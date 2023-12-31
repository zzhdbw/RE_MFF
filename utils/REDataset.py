### REDataset
# -- coding: utf-8 --**
# @author zjp
# @time 22-04-25_20.51.18

import numpy as np
import torch
from torch.utils.data import Dataset

class REDataset(Dataset):
    def __init__(self, data, args):
        self.data = data
        self.max_word_len = args.max_word_len
        self.dep_type = args.dep_type

    def __len__(self):
        return len(self.data)    

    def __getitem__(self, index):  
        input_ids = torch.tensor(self.data[index]['input_ids'] , dtype=torch.long)
        attention_masks = torch.tensor(self.data[index]['attention_masks'] , dtype=torch.long)
        segment_ids = torch.tensor(self.data[index]['segment_ids'] , dtype=torch.long)

        entity_token_ids = torch.tensor(self.data[index]['entity_token_ids'] , dtype=torch.long)
        label_id = torch.tensor(self.data[index]['label_id'] , dtype=torch.long)
        word_feature = torch.tensor(self.data[index]['word_feature'] , dtype=torch.long)
        e1_mask = torch.tensor(self.data[index]['e1_mask'] , dtype=torch.long)
        e2_mask = torch.tensor(self.data[index]['e2_mask'] , dtype=torch.long)

        # print("RE:",self.data[index]['guid'])
        dep_type_matrix = []
        for i in range(len(self.dep_type)):
            dep_mtrix_temp = get_dep_matrix(self.data[index]['dep_type_matrix'][i], self.max_word_len)
            dep_type_matrix.append(dep_mtrix_temp)
        dep_type_matrix = torch.tensor(dep_type_matrix, dtype=torch.long)

        dep_distence_matrix = get_distence_matrix(self.data[index]['dep_distence_matrix'], self.max_word_len)

        return input_ids, attention_masks, segment_ids, entity_token_ids, label_id, word_feature, e1_mask, e2_mask, dep_type_matrix, dep_distence_matrix

def get_dep_matrix(ori_dep_type_matrix, max_word_len):
    '''依存矩阵在此补短'''
    dep_type_matrix = np.zeros((max_word_len, max_word_len), dtype=np.int_)
    max_words_num = len(ori_dep_type_matrix)
    for i in range(max_words_num):
        dep_type_matrix[i][:max_words_num] = ori_dep_type_matrix[i]
    # return torch.tensor(dep_type_matrix, dtype=torch.long)
    return dep_type_matrix

def get_distence_matrix(ori_dep_distence_matrix, max_word_len):
    '''依存距离矩阵在此补短'''
    dep_distence_matrix = np.zeros((max_word_len, max_word_len), dtype=np.float32)
    max_words_num = len(ori_dep_distence_matrix)
    for i in range(max_words_num):
        dep_distence_matrix[i][:max_words_num] = ori_dep_distence_matrix[i]
    return torch.tensor(dep_distence_matrix, dtype=torch.float64)
