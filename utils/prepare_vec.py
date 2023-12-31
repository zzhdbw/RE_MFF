# -- coding: utf-8 --**
# @author zjp
# @time 22-04-25_14.33.35
import torch
import logging
import os
import numpy as np
import fasttext
from tqdm import tqdm

logger = logging.getLogger('root')


class prepare_vector():
    '''获取词索引表, embedding矩阵'''

    def __init__(self, args=None) -> None:
        logger.info("begining of build vector")
        self.args = args
        self.word2index = {}

    def _read_text(self):
        '''读取文件，获取训练文本词汇'''
        datas = []
        file_path = os.path.join(self.args.datadir, self.args.dataset, 'train.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = line.strip('\n').split('\t')
                if len(item) == 8:  ## standford处理的数据
                    e1, e1_type, e2, e2_type, relation, e_pos, raw_sentence, sentence = item
                    datas.append(sentence)
        return datas

    def _word_count(self, datas):
        '''统计单词出现的频次，并将其降序排列，得出出现频次最多的单词'''
        dic = {}
        for data in datas:
            data_list = data.split()
            for word in data_list:
                word = word.lower()  # 所有单词转化为小写
                if (word in dic):
                    dic[word] += 1
                else:
                    dic[word] = 1
        word_count_sorted = sorted(dic.items(), key=lambda item: item[1], reverse=True)
        return word_count_sorted  ## 词典{('，', 112665),('、', 67212),}

    def word_index(self, vocab_size=0):  ## 在此可通过vocab_size控制词表大小
        '''创建词表'''
        datas = self._read_text()
        word_count_sorted = self._word_count(datas)
        word2index = {}
        # 词表中未出现的词
        word2index["<unk>"] = 0
        # 句子添加的padding
        word2index["<pad>"] = 1

        # 词表的实际大小由词的数量和限定大小决定
        # vocab_size = min(len(word_count_sorted), vocab_size)
        vocab_size = len(word_count_sorted)
        for i in range(vocab_size):
            word = word_count_sorted[i][0]
            word2index[word] = i + 2

        self.word2index = word2index
        return word2index

    def get_embedding_matrix(self):
        logger.info("由{}创建embedding嵌入矩阵".format(self.args.vector_file))
        # print("加载词向量文件......")

        fasttext_model = fasttext.load_model(self.args.vector_file)

        # embedding_matrix = np.zeros((len(self.word2index), self.args.vector_dim))
        embedding_matrix = np.random.random((len(self.word2index), self.args.vector_dim))  ## 给["<unk>"]赋随机值
        for word, i in tqdm(self.word2index.items(), desc="创建embedding嵌入矩阵:"):
            if i == 0:  ## ["<unk>"]赋随机值，不使用预训练的词
                continue
            elif i == 1:  ## ["<pad>"]赋值为零
                embedding_matrix[i] = np.zeros(self.args.vector_dim)
            else:
                embedding_vector = fasttext_model.get_word_vector(str(word))
                embedding_matrix[i] = embedding_vector
        return torch.FloatTensor(embedding_matrix)


# from main import get_args, set_params, base_function

if __name__ == "__main__":
    # args = get_args()
    # logger = set_params(args=args)
    # config, tokenizer, dataset = base_function(args, logger)

    fasttext_model = fasttext.load_model("../vector/cc.zh.300.bin")
    embedding_vector = fasttext_model.get_word_vector(str("上山"))

    print(fasttext_model)