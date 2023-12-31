# -- coding: utf-8 --**
# @author zjp
# @time 22-04-24_13.27.02
import os
import torch
import torch.nn as nn
from transformers import BertModel


class Model(nn.Module):
    '''just bert'''

    def __init__(self, args, config):
        super(Model, self).__init__()
        bert_path = os.path.join(args.bert_path, args.bert_name)
        self.bert = BertModel.from_pretrained(bert_path, config=config)
        self.bert.resize_token_embeddings(args.tokenizer_len)  ##因为添加了新的特殊符号
        for param in self.bert.parameters():
            param.requires_grad = True
        self.num_labels = config.num_relation_labels

        self.dropout_bert =nn.Dropout(config.hidden_dropout_prob)
        #         self.classifier = nn.Linear(config.hidden_size * 5, self.num_labels)
        self.classifier = nn.Linear(config.hidden_size * 4, self.num_labels)

    # def forward(self, input_ids, attention_masks, segment_ids, entity_token_ids=None, label_id=None, word_feature=None, e1_mask=None, e2_mask=None, dep_type_matrix=None):
    def forward(self, input_ids, attention_masks, segment_ids, entity_token_ids=None,
                label_id=None, word_feature=None, e1_mask=None, e2_mask=None,
                dep_type_matrix=None, dep_distence_matrix=None):

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
            [torch.index_select(sequence_output[i, :, :], 0, entity_token_ids[i, :]).unsqueeze(0) for i in range(batch_size)], 0)
        # print(e_token.shape)  ##[batchsize, 4, 768]
        e_token = e_token.contiguous().view(batch_size, -1)

        #         out = torch.cat([pooled_output, e_token], dim=-1)
        out = e_token
        out = self.dropout_bert(out)
        logits = self.classifier(out)

        if label_id is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), label_id.view(-1))
            return loss
        else:
            return logits
