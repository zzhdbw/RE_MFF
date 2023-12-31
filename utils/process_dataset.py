# -- coding: utf-8 --**
# @author zjp
# @time 22-04-18_13.21.56
import json
import os
import numpy as np
from tqdm import tqdm
from utils.dep_parser import DepInstanceParser

import logging

logger = logging.getLogger('root')


class process_dataset():
    '''处理数据'''

    def __init__(self, args=None) -> None:
        logger.info("begining of build dataset")
        self.args = args
        self.direct = True  ## 依存关系是否区分方向
        self.dep_type = args.dep_type  ## 依存关系的形式
        # self.dep_type = "_".join(args.dep_type)  ## 依存关系的形式
        self.dataset = args.dataset
        # self.type_dict = {}
        # self.labels_dict = {}

    def get_train_examples(self):
        '''获取训练数据'''
        all_train_feature = self._read_features(data_type="train")
        return self._create_examples(all_train_feature, data_type="train")

    def get_dev_examples(self):
        '''获取验证数据'''
        all_dev_feature = self._read_features(data_type="dev")
        return self._create_examples(all_dev_feature, data_type="dev")

    def get_test_examples(self):
        '''获取测试数据'''
        all_test_feature = self._read_features(data_type="test")
        return self._create_examples(all_test_feature, data_type="test")

    def _find_enyity_range(self, tokens, e1_ids, e2_ids):
        '''通过实体在句子中的位置，找到实体在分词后句子中的位置'''
        temp = ""
        left = 0
        right = 0
        token_ids = []
        e1_range = []
        e2_range = []
        for token in tokens:
            temp = temp + token
            right = len(temp)
            ids = list(range(left, len(temp)))
            token_ids.append(ids)
            left = right
        for i, ids in enumerate(token_ids):
            if ids[0] in e1_ids or ids[-1] in e1_ids:
                e1_range.append(i)
            if ids[0] in e2_ids or ids[-1] in e2_ids:
                e2_range.append(i)
        return e1_range, e2_range

    def _read_features(self, data_type):
        '''获取文本信息和依存树信息, 并进行处理'''
        error = 0
        error_key = []
        text_file_path = os.path.join(self.args.datadir, self.args.dataset, "{}.txt".format(data_type))
        dep_file_path = os.path.join(self.args.datadir, self.args.dataset, "{}.txt.dep".format(data_type))
        all_text_data = self._load_textfile(text_file_path)
        all_dep_info = self._load_depfile(dep_file_path)
        all_feature_data = []
        ## 为加速计算，将处理后的距离矩阵进行存储
        data_file = os.path.join(self.args.datadir, "graph_file",
                                 str(self.dataset) + "_" + str("_".join(self.dep_type)) + "_" + str(data_type) + ".npz")
        if os.path.exists(data_file) and "distance_graph" in self.dep_type:
            logger.info("读取预处理的距离矩阵的文件:" + str(data_file))
            distance_graph_data = np.load(data_file, allow_pickle=True)
            all_dep_adj_matrix = distance_graph_data["all_dep_adj_matrix"]
            i = 0
        else:
            all_dep_adj_matrix = []
        for text_data, dep_info in tqdm(zip(all_text_data, all_dep_info), desc="process {} data".format(data_type)):
            # for ids, (text_data, dep_info) in enumerate(zip(all_text_data, all_dep_info)):
            label = text_data["relation"]

            ori_sentence = text_data['raw_sentence']
            tokens = text_data['sentence'].split(" ")
            e1 = text_data["e1"].split(" ")
            e2 = text_data["e2"].split(" ")

            ''' 问题：返回的实体位置应该是实体在分完词后的列表中的位置
                实体在原句子中的位置可以通过实体标记寻找(已解决_find_enyity_range)'''
            e_pos = text_data['e_pos'].strip("()").split(",")
            e11_p, e12_p, e21_p, e22_p = int(e_pos[0]), int(e_pos[1]), int(e_pos[2]), int(e_pos[3])
            e1_ids = list(range(e11_p, e12_p))
            e2_ids = list(range(e21_p, e22_p))
            e1_range, e2_range = self._find_enyity_range(tokens, e1_ids, e2_ids)  ## 实体范围，应为实体在分词后的句子中的实体位置
            # e1_range = self._find_enyity_range(tokens,e1)
            # e2_range = self._find_enyity_range(tokens,e2)

            try:
                start_range = list(range(e1_range[0], e1_range[-1] + 1))  ## 实体范围，应为实体在分词后的句子中的实体位置
                end_range = list(range(e2_range[0], e2_range[-1] + 1))
            except IndexError:
                # error_key.append((ids,e1))
                # error += 1
                continue
            dep_type_matrix_mix = []
            dep_adj_matrix_dis = []
            dep_instance_parser = DepInstanceParser(basicDependencies=dep_info, tokens=tokens)
            if "_".join(self.dep_type) == "first_order" or "_".join(self.dep_type) == "full_graph":
                dep_adj_matrix, dep_type_matrix = dep_instance_parser.get_first_order(direct=self.direct)
                dep_type_matrix_mix.append(dep_type_matrix)
            elif "_".join(self.dep_type) == "local_graph":
                dep_adj_matrix, dep_type_matrix = dep_instance_parser.get_local_graph(start_range, end_range,
                                                                                      direct=self.direct)
                dep_type_matrix_mix.append(dep_type_matrix)
            elif "_".join(self.dep_type) == "global_graph":
                dep_adj_matrix, dep_type_matrix = dep_instance_parser.get_global_graph(start_range, end_range,
                                                                                       direct=self.direct)
                dep_type_matrix_mix.append(dep_type_matrix)
            elif "_".join(self.dep_type) == "local_global_graph":
                dep_adj_matrix, dep_type_matrix = dep_instance_parser.get_local_global_graph(start_range, end_range,
                                                                                             direct=self.direct)
                dep_type_matrix_mix.append(dep_type_matrix)
            elif "_".join(self.dep_type) == "distance_graph":
                if not os.path.exists(data_file):
                    dep_adj_matrix_dis, dep_type_matrix = dep_instance_parser.get_distance_graph(self.args,
                                                                                                 direct=self.direct)
                    all_dep_adj_matrix.append(dep_adj_matrix_dis)
                    # all_dep_type_matrix.append(dep_type_matrix)   
                else:
                    dep_adj_matrix_dis = all_dep_adj_matrix[i]
                    # dep_type_matrix = all_dep_adj_matrix[i]   ## 因为 dep_type_matrix 无法存储的问题，就不存了，反正在有距离时，不通过get_distance_graph获取dep_type_matrix
                    _, dep_type_matrix = dep_instance_parser.get_init_dep_matrix()
                    i = i + 1
                dep_type_matrix_mix.append(dep_type_matrix)
            elif "_".join(self.dep_type) == 'distance_graph_local_graph':
                if not os.path.exists(data_file):  ## 距离图不需要类型矩阵，，非距离图只需要类型矩阵
                    dep_adj_matrix_dis, dep_type_matrix_dis = dep_instance_parser.get_distance_graph(self.args,
                                                                                                     direct=self.direct)
                    all_dep_adj_matrix.append(dep_adj_matrix_dis)
                    # all_dep_type_matrix.append(dep_type_matrix)   
                else:
                    dep_adj_matrix_dis = all_dep_adj_matrix[i]
                    # dep_type_matrix = all_dep_adj_matrix[i]   ## 因为 dep_type_matrix 无法存储的问题，就不存了，反正在有距离时，不通过get_distance_graph获取dep_type_matrix
                    _, dep_type_matrix_dis = dep_instance_parser.get_init_dep_matrix()
                    i = i + 1
                dep_adj_matrix, dep_type_matrix_local = dep_instance_parser.get_local_graph(start_range, end_range,
                                                                                            direct=self.direct)
                dep_type_matrix_mix.append(dep_type_matrix_dis)
                dep_type_matrix_mix.append(dep_type_matrix_local)
            elif "_".join(self.dep_type) == 'distance_graph_global_graph':
                if not os.path.exists(data_file):  ## 距离图不需要类型矩阵，，非距离图只需要类型矩阵
                    dep_adj_matrix_dis, dep_type_matrix_dis = dep_instance_parser.get_distance_graph(self.args,
                                                                                                     direct=self.direct)
                    all_dep_adj_matrix.append(dep_adj_matrix_dis)
                    # all_dep_type_matrix.append(dep_type_matrix)   
                else:
                    dep_adj_matrix_dis = all_dep_adj_matrix[i]
                    # dep_type_matrix = all_dep_adj_matrix[i]   ## 因为 dep_type_matrix 无法存储的问题，就不存了，反正在有距离时，不通过get_distance_graph获取dep_type_matrix
                    _, dep_type_matrix_dis = dep_instance_parser.get_init_dep_matrix()
                    i = i + 1
                dep_adj_matrix, dep_type_matrix_global = dep_instance_parser.get_global_graph(start_range, end_range,
                                                                                              direct=self.direct)
                dep_type_matrix_mix.append(dep_type_matrix_dis)
                dep_type_matrix_mix.append(dep_type_matrix_global)
            elif "_".join(self.dep_type) == 'distance_graph_local_graph_global_graph':
                if not os.path.exists(data_file):  ## 距离图不需要类型矩阵，，非距离图只需要类型矩阵
                    dep_adj_matrix_dis, dep_type_matrix_dis = dep_instance_parser.get_distance_graph(self.args,
                                                                                                     direct=self.direct)
                    all_dep_adj_matrix.append(dep_adj_matrix_dis)
                    # all_dep_type_matrix.append(dep_type_matrix)   
                else:
                    dep_adj_matrix_dis = all_dep_adj_matrix[i]
                    # dep_type_matrix = all_dep_adj_matrix[i]   ## 因为 dep_type_matrix 无法存储的问题，就不存了，反正在有距离时，不通过get_distance_graph获取dep_type_matrix
                    _, dep_type_matrix_dis = dep_instance_parser.get_init_dep_matrix()
                    i = i + 1
                dep_adj_matrix, dep_type_matrix_local = dep_instance_parser.get_local_graph(start_range, end_range,
                                                                                            direct=self.direct)
                dep_adj_matrix, dep_type_matrix_global = dep_instance_parser.get_global_graph(start_range, end_range,
                                                                                              direct=self.direct)
                dep_type_matrix_mix.append(dep_type_matrix_dis)
                dep_type_matrix_mix.append(dep_type_matrix_local)
                dep_type_matrix_mix.append(dep_type_matrix_global)
            all_feature_data.append({
                "words": dep_instance_parser.words,  ##组成句子的词，列表
                "ori_sentence": ori_sentence,
                "dep_adj_dis_matrix": dep_adj_matrix_dis,  ## dep_adj_dis_matrix 在不需要距离的图中没用到，因此只用于存放距离图
                "dep_type_matrix": dep_type_matrix_mix,  ## 存放非距离图的依存类型矩阵，后去可有依存类型矩阵夹逼出依存矩阵clamp
                "label": label,
                "e1": tokens[e1_range[0]: e1_range[-1] + 1],  ##text_data["e1"],
                "e2": tokens[e2_range[0]: e2_range[-1] + 1],  ##text_data["e2"],
                "e_pos": (e1_range[0], e1_range[-1] + 1, e2_range[0], e2_range[-1] + 1)
            })
        # print(error)
        # print(error_key)
        ##  存储带距离的邻接矩阵
        if not os.path.exists(os.path.join(self.args.datadir, "graph_file")):
            os.makedirs(os.path.join(self.args.datadir, "graph_file"))
        if not os.path.exists(data_file) and "distance" in "_".join(self.dep_type):
            all_dep_adj_matrix = np.array(all_dep_adj_matrix)
            np.savez(data_file,
                     all_dep_adj_matrix=all_dep_adj_matrix)
        return all_feature_data

    def _create_examples(self, features, data_type):
        '''添加数据类型, 和编号'''
        examples = []
        for i, feature in enumerate(features):
            guid = "%s-%s" % (data_type, i)
            feature["guid"] = guid
            examples.append(feature)
        return examples

    def _load_textfile(self, file_path):
        '''加载文本信息'''
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:  ## LTP处理的数据
                item = line.strip('\n').split('\t')
                if len(item) == 7:
                    e1, e1_type, e2, e2_type, relation, raw_sentence, sentence = item
                    data.append({
                        "e1": e1,
                        "e1_type": e1_type,
                        "e2": e2,
                        "e2_type": e2_type,
                        "relation": relation,
                        "raw_sentence": raw_sentence,
                        "sentence": sentence
                    })
                elif len(item) == 8:  ## standford处理的数据
                    e1, e1_type, e2, e2_type, relation, e_pos, raw_sentence, sentence = item
                    data.append({
                        "e1": e1,
                        "e1_type": e1_type,
                        "e2": e2,
                        "e2_type": e2_type,
                        "relation": relation,
                        'e_pos': e_pos,
                        "raw_sentence": raw_sentence,
                        "sentence": sentence
                    })
        return data

    def _load_depfile(self, file_path):
        '''读取依存树信息文件'''
        data = []
        with open(file_path, 'r', encoding="utf-8") as f:
            dep_info = []
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    items = line.split("\t")
                    dep_info.append({
                        "governor": int(items[0]),
                        "dependent": int(items[1]),
                        "dep": items[2],
                    })
                else:  ## 一个句子的依存树读取完毕
                    if len(dep_info) > 0:
                        data.append(dep_info)
                        dep_info = []
            if len(dep_info) > 0:
                data.append(dep_info)
                dep_info = []
        return data

    def _get_dep_labels(self, dep_type_path):
        '''读取dep_type.json文件'''
        dep_labels = ["self_loop"]
        with open(dep_type_path, 'r', encoding='utf-8') as f:
            dep_types = json.load(f)
            for label in dep_types:
                if self.direct:
                    dep_labels.append("{}_in".format(label))
                    dep_labels.append("{}_out".format(label))
                else:
                    dep_labels.append(label)
        return dep_labels

    def prepare_type_dict(self):
        '''获取依存类型字典'''
        dep_type_path = os.path.join(self.args.datadir, self.args.dataset, "dep_type.json")
        dep_type_list = self._get_dep_labels(dep_type_path)
        types_dict = {"none": 0}
        for dep_type in dep_type_list:
            types_dict[dep_type] = len(types_dict)
        # self.types_dict = types_dict
        return types_dict

    def prepare_labels_dict(self):
        '''获取标签字典'''
        label_path = os.path.join(self.args.datadir, self.args.dataset, "label.json")
        labels_dict = {}
        with open(label_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        for label in labels:
            labels_dict[label] = len(labels_dict)
            # self.labels_dict = labels_dict
        return labels_dict
