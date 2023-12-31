# -- coding: utf-8 --**
# @author zjp
# @time 22-11-05_09.49.22
import os 
import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel
# from models.gcn_type import GraphConvolution
from models.gcn_base import GraphConvolution
'''局部连接加上距离矩阵'''
class Model(nn.Module):
    '''bert_BiLSTM_typegcn'''
    def __init__(self, args, config):
        super(Model, self).__init__()
        bert_path = os.path.join(args.bert_path, args.bert_name)
        self.bert = BertModel.from_pretrained(bert_path, config=config)
        self.bert.resize_token_embeddings(args.tokenizer_len)##因为添加了新的特殊符号
        for param in self.bert.parameters():
            param.requires_grad = True

        if len(args.embedding_matrix) > 0 :
            self.words_embedding = nn.Embedding.from_pretrained(embeddings=args.embedding_matrix, freeze=True)   ## Freeze=True,表示这层不参与训练
        else:
            self.words_embedding = nn.Embedding(num_embeddings=len(args.word2index), embedding_dim=args.vector_dim, padding_idx=1) ## pad 位置为零        
        self.lstm_word = nn.LSTM(input_size = args.vector_dim,
                            hidden_size = args.grnn_dim,    ## 128
                            num_layers = 1, bidirectional = True, batch_first = True)                            
        self.num_labels = config.num_relation_labels
        self.renorm = args.renorm   ## 重整化参数
        self.gcn_layer_dis1 = GraphConvolution(args.grnn_dim*2, args.grnn_dim*2,self.renorm)
        self.gcn_layer_dis2 = GraphConvolution(args.grnn_dim*2, args.grnn_dim*2,self.renorm)
        self.gcn_layer_global1 = GraphConvolution(args.grnn_dim*2, args.grnn_dim*2,self.renorm)
        self.gcn_layer_global2 = GraphConvolution(args.grnn_dim*2, args.grnn_dim*2,self.renorm)
        self.dropout_bert = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(args.grnn_dim * 2 * 4 + config.hidden_size * 5, self.num_labels)
    
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


    def forward(self, input_ids, attention_masks, segment_ids, entity_token_ids=None, 
                label_id=None, word_feature=None, e1_mask=None, e2_mask=None, dep_type_matrix=None, dep_distence_matrix=None):
        bert_outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_masks,
            token_type_ids = segment_ids,
        )
        sequence_output = bert_outputs[0]   ## [batchsize, max_len, 768]
        pooled_output = bert_outputs[1]    ## [batchsize, 768]
        batch_size = sequence_output.size(0)
        sequence_output = self.dropout_bert(sequence_output)
        e_token = torch.cat([torch.index_select(sequence_output[i,:,:],0,entity_token_ids[i,:]).unsqueeze(0) for i in range(batch_size)], 0)
        e_token = e_token.contiguous().view(batch_size, -1)
        bert_out = torch.cat([pooled_output, e_token], dim=-1)

        word_embedding = self.words_embedding(word_feature)

        max_word_len = word_embedding.size(1)
        word_out, (h_n, c_n) = self.lstm_word(word_embedding) ## [batch_size, max_word_len, rnn_dim *2]        
        word_out = self.valid_filter(word_out, word_feature)

        ## dep_type_matrix以数组形式存储 dis和 global
        dep_type_matrix_dis = dep_type_matrix[:,0,:,:]   ## 距离矩阵获取的依存类型图为全图
        dep_type_matrix_global = dep_type_matrix[:,1,:,:]   
        dep_adj_matrix_global = torch.clamp(dep_type_matrix_global, min=0, max=1)

        word_out_global = self.gcn_layer_global1(word_out, dep_adj_matrix_global)
        word_out_global = self.gcn_layer_global2(word_out_global, dep_adj_matrix_global)    ## [1, 80, 256]  ## e1_mask [1, 80]
        e1_h_global = self.extract_entity(word_out_global, e1_mask)       ## [1, 256]
        e2_h_global = self.extract_entity(word_out_global, e2_mask)

        word_out_dis = self.gcn_layer_dis1(word_out, dep_distence_matrix)
        word_out_dis = self.gcn_layer_dis2(word_out_dis, dep_distence_matrix)

        e1_h_dis = self.extract_entity(word_out_dis, e1_mask)       ## [1, 256]
        e2_h_dis = self.extract_entity(word_out_dis, e2_mask)

        out = torch.cat([e1_h_global,e2_h_global,e1_h_dis,e2_h_dis,bert_out],dim=-1)  ## 4864
        logits = self.classifier(out)

        if label_id is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), label_id.view(-1))
            return loss
        else:
            return logits