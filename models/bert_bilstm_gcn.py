# -- coding: utf-8 --**
# @author zjp
# @time 2022-08-31_18.06.24
import os
import torch
import torch.nn as nn
from transformers import BertModel
# from models.gcn_type import GraphConvolution
from models.gcn_base import GraphConvolution

'''最终向量,提取实体位置向量作为分类器的输入'''


class Model(nn.Module):
    '''bert_BiLSTM_typegcn'''

    def __init__(self, args, config):
        super(Model, self).__init__()
        bert_path = os.path.join(args.bert_path, args.bert_name)
        self.bert = BertModel.from_pretrained(bert_path, config=config)
        self.bert.resize_token_embeddings(args.tokenizer_len)  ##因为添加了新的特殊符号
        for param in self.bert.parameters():
            param.requires_grad = True

        if len(args.embedding_matrix) > 0:
            self.words_embedding = nn.Embedding.from_pretrained(embeddings=args.embedding_matrix,
                                                                freeze=True)  ## Freeze=True,表示这层不参与训练
        else:
            self.words_embedding = nn.Embedding(num_embeddings=len(args.word2index), embedding_dim=args.vector_dim,
                                                padding_idx=1)  ## pad 位置为零
        self.dep_type_embedding = nn.Embedding(config.type_num, args.grnn_dim * 2,
                                               padding_idx=0)  ## padding_idx=0因为依存关系类中添加了无类型的none，要初始化为0
        self.lstm_word = nn.LSTM(input_size=args.vector_dim,
                                 hidden_size=args.grnn_dim,  ## 128
                                 num_layers=1, bidirectional=True, batch_first=True)
        self.num_labels = config.num_relation_labels
        self.renorm = args.renorm  ## 重整化参数
        self.gcn_layer1 = GraphConvolution(args.grnn_dim * 2, args.grnn_dim * 2, self.renorm)
        self.gcn_layer2 = GraphConvolution(args.grnn_dim * 2, args.grnn_dim * 2, self.renorm)
        self.dropout_bert = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(args.grnn_dim * 2 * 2 + config.hidden_size * 5, self.num_labels)

    def valid_filter(self, word_out, word_feature):
        '''根据word_feature替换padding获取的向量, 替换为零, word_feature中padding位置填充为1'''
        batch_size, max_len, hidden_size = word_out.shape
        valid_out = torch.zeros([batch_size, max_len, hidden_size], dtype=word_out.dtype, device=word_out.device)
        for i in range(batch_size):
            temp = word_out[i][word_feature[i] != 1]
            valid_out[i][:temp.size(0)] = temp
        return valid_out

    def extract_entity(self, word_out, e_mask):
        '''取实体位置的值'''
        ## torch.stack在指定维度复制列表
        entity_output = word_out * torch.stack([e_mask] * word_out.shape[-1], dim=2) + torch.stack(
            [(1.0 - e_mask) * -1000.0] * word_out.shape[-1], 2)
        entity_output = torch.max(entity_output, -2)[0]
        return entity_output.type_as(word_out)

    def get_attention(self, word_out, dep_embed, adj):
        '''
        使用注意力机制, 对邻接矩阵进行加权, 使用关系类型对获取注意力概率分布
        ## dep_embed 是e^t_{i,j}
        '''
        batch_size, max_len, feat_dim = word_out.shape  ## dep_embed 是e^t_{i,j}
        val_us = word_out.unsqueeze(dim=2)
        val_us = val_us.repeat(1, 1, max_len, 1)  ##将句子的每个词复制句子长度次
        val_cat = torch.cat((val_us, dep_embed), -1)  ## s_i,, ## 为每个词添加该词与其他词之间的依存类型
        atten_expand = (val_cat.float() * val_cat.float().transpose(1,
                                                                    2))  ##s_i*s_j即每个节点都与其他节点乘了一次 ## val_cat为s_i,反转后为s_j ## 点积
        attention_score = torch.sum(atten_expand, dim=-1)  ## 将每个词的隐藏维度求和，表示当前边的权重
        attention_score = attention_score / feat_dim ** 0.5  ## 缩放
        # softmax
        exp_attention_score = torch.exp(attention_score)
        exp_attention_score = torch.mul(exp_attention_score.float(), adj.float())  ## 求注意力概率期间，没关系的边也被赋予的权重，在此剔除
        sum_attention_score = torch.sum(exp_attention_score, dim=-1).unsqueeze(dim=-1).repeat(1, 1,
                                                                                              max_len)  ## 求每个词所有关系的总注意力概率和（每一行的注意力概率和），并重复maxlen次
        attention_score = torch.div(exp_attention_score, sum_attention_score + 1e-10)
        return attention_score

    def forward(self, input_ids, attention_masks, segment_ids, entity_token_ids=None, label_id=None,
                word_feature=None, e1_mask=None, e2_mask=None, dep_type_matrix=None, dep_distence_matrix=None):
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_masks,
            token_type_ids=segment_ids,
        )
        sequence_output = bert_outputs[0]  ## [batchsize, max_len, 768]
        pooled_output = bert_outputs[1]  ## [batchsize, 768]

        batch_size = sequence_output.size(0)
        sequence_output = self.dropout_bert(sequence_output)
        e_token = torch.cat(
            [torch.index_select(sequence_output[i, :, :], 0, entity_token_ids[i, :]).unsqueeze(0) for i in
             range(batch_size)], 0)
        e_token = e_token.contiguous().view(batch_size, -1)
        bert_out = torch.cat([pooled_output, e_token], dim=-1)

        word_embedding = self.words_embedding(word_feature)

        max_word_len = word_embedding.size(1)
        word_out, (h_n, c_n) = self.lstm_word(word_embedding)  ## [batch_size, max_word_len, rnn_dim *2]
        word_out = self.valid_filter(word_out, word_feature)

        # dep_type_embedding_ouput = self.dep_type_embedding(dep_type_matrix) ## [1, 80, 80, 300]
        dep_type_matrix = dep_type_matrix[:, 0, :, :]
        dep_adj_matrix = torch.clamp(dep_type_matrix, min=0, max=1)  ## [1, 80, 80]
        if self.renorm != 0:  ##使用重整化参数
            diag = torch.full(size=[batch_size, max_word_len], fill_value=self.renorm - 1,
                              device=dep_type_matrix.device)  ## 邻接矩阵本就带着自身节点的链接，所以要先减去1
            diag = torch.diag_embed(diag)  ## 生成带重整化参数的单位矩阵
            dep_adj_matrix = dep_adj_matrix + diag  ## adj + diag为A+yI，，sum求和得到度矩阵D+yI

        # attention_score = self.get_attention(word_out, dep_type_embedding_ouput, dep_adj_matrix)
        word_out = self.gcn_layer1(word_out, dep_adj_matrix)
        word_out = self.gcn_layer2(word_out, dep_adj_matrix)  ## [1, 80, 256]  ## e1_mask [1, 80]
        e1_h = self.extract_entity(word_out, e1_mask)  ## [1, 256]
        e2_h = self.extract_entity(word_out, e2_mask)

        out = torch.cat([e1_h, e2_h, bert_out], dim=-1)
        logits = self.classifier(out)

        if label_id is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), label_id.view(-1))
            return loss
        else:
            return logits
