# -- coding: utf-8 --**
# @author zjp
# @time 22-04-24_15.19.39
from utils.sdp import get_adj_sdp
from tqdm import tqdm
import numpy as np
import logging

logger = logging.getLogger('root')


def build_dataset(examples, tokenizer, word2index, args):
    '''使用BERT tokenizer 处理数据集'''
    dep_label_map = args.types_dict
    label_map = args.labels_dict
    maxlen = args.maxlen  ## 最大句子长度
    max_word_len = args.max_word_len  ## 分词后最大句子长度
    miss_token_seq_num = 0  ## 因句子截长补短而找不到的标记的句子的数量
    features = []
    flag = False  ##写日志的标记
    for example in tqdm(examples, desc="bulid dataset"):
        # for ids, example in enumerate(examples):
        ### 获取bert需要的内容
        ori_sentence = example['ori_sentence']
        entity_token_ids = []  ## 实体标记所在位置
        tokens = tokenizer.tokenize(ori_sentence)
        seq_len = len(tokens)
        encode_dict = tokenizer.encode_plus(
            text=tokens,
            add_special_tokens=True,  ##
            max_length=maxlen,
            truncation='longest_first',
            padding="max_length",
            return_token_type_ids=True,
            return_attention_mask=True)

        input_ids = encode_dict['input_ids']
        attention_masks = encode_dict['attention_mask']
        token_type_ids = encode_dict['token_type_ids']
        ## 找出实体标记，并按照头尾实体的顺序排列
        entity_token_list = [et for et in tokens if et in args.ADDITIONAL_SPECIAL_TOKENS]
        try:  ## 找到标记的位置，若不存在则用第一个[CLS]表示，此标记表示token后实体的位置，因为token是截长，有些超长的就被裁剪了，所以需要此操作
            for special_token in entity_token_list:
                special_token_ids = input_ids.index(tokenizer.convert_tokens_to_ids(special_token))
                entity_token_ids.append(special_token_ids)
        except ValueError:
            entity_token_ids.extend([0] * (4 - len(entity_token_ids)))
            miss_token_seq_num += 1
        if len(tokens) > maxlen:
            seq_len = maxlen

        assert len(input_ids) == maxlen
        assert len(attention_masks) == maxlen
        assert len(token_type_ids) == maxlen

        #### 获取LSTM需要的内容
        label_id = label_map[example['label']]  ## 获取关系标签
        e_pos = example['e_pos']
        e1_mask, e2_mask = [], []
        word_feature = []  ## 分词的句子的特征
        for word_ids, word in enumerate(example['words']):
            word = word.lower()  ## 将其中的英文单词小写
            if word in word2index:
                word_feature.append(word2index[word])
            else:
                word_feature.append(word2index['<unk>'])
            if word_ids >= e_pos[0] and word_ids < e_pos[1]:
                e1_mask.append(1)
            else:
                e1_mask.append(0)
            if word_ids >= e_pos[2] and word_ids < e_pos[3]:
                e2_mask.append(1)
            else:
                e2_mask.append(0)
            if len(word_feature) == max_word_len:
                break
        max_words_num = len(word_feature)  ## 真实句子特征的长度
        word_feature = word_feature + [word2index['<pad>']] * (max_word_len - len(word_feature))
        e1_mask = e1_mask + [0] * (max_word_len - len(e1_mask))
        e2_mask = e2_mask + [0] * (max_word_len - len(e2_mask))

        assert len(word_feature) == max_word_len
        assert len(e2_mask) == max_word_len
        assert len(e2_mask) == max_word_len
        ## 获取GCN所需要的内容
        # max_words_num = len(example['words'])   ## 真实句子长度
        # dep_type_matrix = get_adj_with_value_matrix(example['dep_type_matrix'], max_words_num, max_word_len, dep_label_map)
        dep_type_matrix = [get_adj_with_value_matrix(example['dep_type_matrix'][i],  ##
                                                     max_words_num,
                                                     max_word_len,
                                                     dep_label_map)
                           for i in range(len(args.dep_type))]
        # assert len(dep_type_matrix[0]) == max_word_len or len(dep_type_matrix[0]) == max_words_num

        dep_distence_matrix = []
        if "distance_graph" in args.dep_type:  ## 当需要距离矩阵时
            dep_distence_matrix = get_dis_with_value_matrix(example['dep_adj_dis_matrix'], max_words_num, max_word_len)

        # assert len(dep_distence_matrix) == max_word_len or len(dep_distence_matrix) == max_words_num

        features.append({
            "guid": example["guid"],
            "input_ids": input_ids,
            "attention_masks": attention_masks,
            "segment_ids": token_type_ids,
            "entity_token_ids": entity_token_ids,
            "label_id": label_id,
            "word_feature": word_feature,
            "e1_mask": e1_mask,
            "e2_mask": e2_mask,
            "dep_type_matrix": dep_type_matrix,
            "dep_distence_matrix": dep_distence_matrix
        })
        if flag:
            logger.info("guid:{}".format(example["guid"]))
            logger.info("ori_sentence:{}".format(example["ori_sentence"]))
            logger.info("label:{} label_id:{}".format(example['label'], label_id))
            logger.info("tokens:{}".format(tokens))
            logger.info("input_ids:{}".format(input_ids))
            logger.info("entity_token_ids:{}".format(entity_token_ids))
            logger.info("words:{}".format(example['words']))
            logger.info("word_feature:{}".format(word_feature))
            logger.info("e_pos:{}".format(example['e_pos']))
            logger.info("e1_mask:{}".format(e1_mask))
            logger.info("e2_mask:{}".format(e2_mask))
            # logger.info("dep_type_matrix:\n{}".format(" ".join([str(x) for x in dep_type_matrix])))
            # logger.info("dep_type_matrix:\n{}".format(" ".join([str(x) for x in example['dep_type_matrix']])))
            flag = False
    logger.info(f"找不到标记的句子数量: {miss_token_seq_num}")
    return features


def get_adj_with_value_matrix(dep_type_matrix, max_words_num, max_word_len, dep_label_map):
    ''' 将依存类型矩阵转换为数字，向量化, 依存矩阵在此截长
        不处理依存矩阵的原因为, 依存矩阵可以由依存类型矩阵获得, 使用torch.clamp'''
    final_dep_type_matrix = np.zeros((max_words_num, max_words_num), dtype=np.int_)  ##
    for pi in range(max_words_num):
        for pj in range(max_words_num):
            if dep_type_matrix[pi][pj] == 'none':  ## 在dep_label_map中'none'的索引为0
                continue
            if pi >= max_word_len or pj >= max_word_len:  ## 超长的不要，截长
                continue
            final_dep_type_matrix[pi][pj] = dep_label_map[dep_type_matrix[pi][pj]]

    return final_dep_type_matrix


def get_dis_with_value_matrix(dep_adj_matrix, max_words_num, max_word_len):
    ''' 将距离矩阵向量化, 并在此截长, 在获取distance时, 将距离矩阵存入了dep_adj_matrix'''
    final_dep_adj_matrix = np.zeros((max_words_num, max_words_num), dtype=np.float32)  ##
    for pi in range(max_words_num):
        for pj in range(max_words_num):
            if pi >= max_word_len or pj >= max_word_len:  ## 超长的不要，截长
                continue
            final_dep_adj_matrix[pi][pj] = dep_adj_matrix[pi][pj]
    return final_dep_adj_matrix
