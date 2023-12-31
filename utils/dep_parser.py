# -- coding: utf-8 --**
# @author zjp
# @time 22-04-18_17.39.02
import numpy as np
from utils.sdp import get_adj_sdp
import os 
import networkx as nx
class DepInstanceParser():
    ''' 获取句法依存树, 
        使用standford句法依存工具获得的依存信息生成依存矩阵
        每句话一个依存矩阵'''
    def __init__(self, basicDependencies, tokens=[]) -> None:
        '''初始化句法依存树'''
        self.basicDependencies = basicDependencies  ## 依存关系列表[(root,0, 5)]
        self.tokens = tokens    ## 句子分词列表
        self.words = []
        self.dep_governed_info = []
        self.dep_parsing()

    def dep_parsing(self):
        '''读取依存结构信息, 将依存信息与每个词对应'''
        if len(self.tokens) > 0 :
            words = []
            for token in self.tokens:
                words.append(token)
            dep_governed_info = [       ## 
                {'word': word} for word in words
            ]
            self.words = words
        else:
            dep_governed_info = [{}] * len(self.basicDependencies)
        for dep in self.basicDependencies:
            governed_index = dep['governor'] - 1    ## 出边
            dependent_index = dep['dependent'] - 1  ## 入边
            dep_governed_info[dependent_index] = {  ##讲对应位置的词替换为依存信息
                'governor':governed_index,
                'dep':dep['dep']
                }
        self.dep_governed_info = dep_governed_info

    # def search_dep_path(self, start_idx, end_idx,adj_martix, dep_path_arr):
    #     '''搜索最短依存路径
    #         params:起始节点，结束节点，类型矩阵，节点列表(已包含起始节点)'''
    #     for next_id in range(len(adj_martix[start_idx])):
    #         if next_id in dep_path_arr or adj_martix[start_idx][next_id] in ["none"]:
    #             continue    ## 已添加节点，以及无关系节点 不在判断
    #         if next_id == end_idx:  ## 当前节点为结束节点
    #             return 1, dep_path_arr + [next_id]
    #         ## 递归判断下一节点
    #         stat, dep_arr =self.search_dep_path(next_id, end_idx, adj_martix, dep_path_arr+[next_id])
    #         if stat == 1 :  ## 结束条件
    #             return stat, dep_arr
    #     return 0, []
    def get_init_dep_matrix(self):
        ''' 初始化依存矩阵
            邻接矩阵对角为1, 其余为0
            类型矩阵对角线为"self_loop",其余为"none"'''
        words_len = len(self.words)
        dep_adj_matrix = [[0] * words_len for _ in range(words_len)]
        dep_type_matrix = [['none'] * words_len for _ in range(words_len)]
        for i in range(words_len):
            dep_adj_matrix[i][i] = 1
            dep_type_matrix[i][i] = "self_loop"
        return dep_adj_matrix, dep_type_matrix 

    def get_first_order(self, direct=True, get_dis=False):
        '''获取依存树全图
            存在问题: ROOT节点governor=0,在dep_parsing时更新为governor=-1,
            所以在生成依存关系矩阵时, ROOT根节点被迫与最后一个节点(-1节点)产生关系,
            无论根节点与-1节点是否有关系
            解答: 根节点与-1节点（必为标点符号）必有关系'''
        dep_adj_matrix, dep_type_matrix = self.get_init_dep_matrix()
        for ids, dep_info in enumerate(self.dep_governed_info):
            governed_index = dep_info['governor'] ## 出边，控制词
            dep_type = dep_info['dep']      ## 依存关系
            dep_adj_matrix[ids][governed_index] = 1     ## 存在关系则置位1 
            dep_adj_matrix[governed_index][ids] = 1
            dep_type_matrix[ids][governed_index] = dep_type if direct is False else "{}_in".format(dep_type)
            dep_type_matrix[governed_index][ids] = dep_type if direct is False else "{}_out".format(dep_type)

        return dep_adj_matrix, dep_type_matrix 
    
    def get_local_graph(self, start_range, end_range, direct=True):
        '''实体一个距离的依存关系图 local'''
        dep_path_adj_matrix, dep_path_type_matrix = self.get_init_dep_matrix()
        first_order_dep_adj_matrix, first_order_dep_type_martix = self.get_first_order(direct=direct)

        for start_index in start_range+end_range:
            for next_id in range(len(first_order_dep_type_martix[start_index])):
                if first_order_dep_type_martix[start_index][next_id] in ["none"]:
                    continue    ## 该条边不存在关系
                dep_path_adj_matrix[start_index][next_id] = first_order_dep_adj_matrix[start_index][next_id]
                dep_path_type_matrix[start_index][next_id] = first_order_dep_type_martix[start_index][next_id]
                dep_path_adj_matrix[next_id][start_index] = first_order_dep_adj_matrix[next_id][start_index]
                dep_path_type_matrix[next_id][start_index] = first_order_dep_type_martix[next_id][start_index]
        return dep_path_adj_matrix, dep_path_type_matrix  
    
    # def get_global_graph(self, start_range, end_range, direct=True):  ## 使用原装的方法获取SDP
    #     '''最短依存路径关系图 global'''
    #     dep_path_adj_matrix, dep_path_type_matrix = self.get_init_dep_matrix()
    #     first_order_dep_adj_matrix, first_order_dep_type_martix = self.get_first_order(direct=direct)
    #     for start_index in start_range: ## 每个头尾实体可能有多个字组成
    #         for end_index in end_range:
    #             _, dep_path_indexs = self.search_dep_path(start_index, end_index, first_order_dep_type_martix, [start_index])
    #             for left_index, right_index in zip(dep_path_indexs[:-1], dep_path_indexs[1:]):  ##组合最短依存路径，并遍历
    #                 dep_path_adj_matrix[left_index][right_index] = 1
    #                 dep_path_type_matrix[left_index][right_index] = first_order_dep_type_martix[left_index][right_index]
    #                 dep_path_adj_matrix[right_index][left_index] = 1
    #                 dep_path_type_matrix[right_index][left_index] = first_order_dep_type_martix[right_index][left_index]                    
    #     return dep_path_adj_matrix, dep_path_type_matrix

    # def get_local_global_graph(self, start_range, end_range, direct=True):     ## 使用原装的方法获取SDP
    #     '''实体一个距离的依存关系图和最短依存路径关系图'''
    #     dep_path_adj_matrix, dep_path_type_matrix = self.get_init_dep_matrix()
    #     first_order_dep_adj_matrix, first_order_dep_type_martix = self.get_first_order(direct=direct)
    #     for start_index in start_range+end_range:
    #         for next_id in range(len(first_order_dep_type_martix[start_index])):
    #             if first_order_dep_type_martix[start_index][next_id] in ["none"]:
    #                 continue    ## 该条边不存在关系
    #             dep_path_adj_matrix[start_index][next_id] = first_order_dep_adj_matrix[start_index][next_id]
    #             dep_path_type_matrix[start_index][next_id] = first_order_dep_type_martix[start_index][next_id]
    #             dep_path_adj_matrix[next_id][start_index] = first_order_dep_adj_matrix[next_id][start_index]
    #             dep_path_type_matrix[next_id][start_index] = first_order_dep_type_martix[next_id][start_index]
    #     for start_index in start_range: ## 每个头尾实体可能有多个字组成
    #         for end_index in end_range:
    #             _, dep_path_indexs = self.search_dep_path(start_index, end_index, first_order_dep_type_martix, [start_index])
    #             for left_index, right_index in zip(dep_path_indexs[:-1], dep_path_indexs[1:]):  ##组合最短依存路径，并遍历
    #                 dep_path_adj_matrix[left_index][right_index] = 1
    #                 dep_path_type_matrix[left_index][right_index] = first_order_dep_type_martix[left_index][right_index]
    #                 dep_path_adj_matrix[right_index][left_index] = 1
    #                 dep_path_type_matrix[right_index][left_index] = first_order_dep_type_martix[right_index][left_index]                    
    #     return dep_path_adj_matrix, dep_path_type_matrix
    

    def get_global_graph(self, start_range, end_range, direct=True):  ## 使用dijkstra的方法获取SDP
        '''最短依存路径关系图 global'''
        dep_path_adj_matrix, dep_path_type_matrix = self.get_init_dep_matrix()
        first_order_dep_adj_matrix, first_order_dep_type_martix = self.get_first_order(direct=direct)
        # get_sdp = get_adj_sdp(first_order_dep_adj_matrix) ## v1
        first_order_dep_adj_matrix = np.array(first_order_dep_adj_matrix)   ## v2  v3      
        G = nx.from_numpy_matrix(first_order_dep_adj_matrix)    ## v2   v3
        for start_index in start_range: ## 每个头尾实体可能有多个字组成
            # _, sdps = get_sdp.dijkstra(start_index)   ## v1
            ## v2
            sdps_tuple = list(nx.single_source_dijkstra_path(G,source=start_index).items())
            sdps_tuple = sorted(sdps_tuple, key = lambda k :k[0])
            idx, sdps = zip(*sdps_tuple)
            for end_index in end_range:                
                dep_path_indexs = sdps[end_index]   ## v1 ## v2
                # dep_path_indexs = nx.dijkstra_path(G,source=start_index,target=end_index) ## v3
                # _, dep_path_indexs = self.search_dep_path(start_index, end_index, first_order_dep_type_martix, [start_index]) ## v0
                for left_index, right_index in zip(dep_path_indexs[:-1], dep_path_indexs[1:]):  ##组合最短依存路径，并遍历
                    dep_path_adj_matrix[left_index][right_index] = 1
                    dep_path_type_matrix[left_index][right_index] = first_order_dep_type_martix[left_index][right_index]
                    dep_path_adj_matrix[right_index][left_index] = 1
                    dep_path_type_matrix[right_index][left_index] = first_order_dep_type_martix[right_index][left_index]                    
        return dep_path_adj_matrix, dep_path_type_matrix

    def get_local_global_graph(self, start_range, end_range, direct=True):     ## 使用dijkstra的方法获取SDP
        '''实体一个距离的依存关系图和最短依存路径关系图'''
        dep_path_adj_matrix, dep_path_type_matrix = self.get_init_dep_matrix()
        first_order_dep_adj_matrix, first_order_dep_type_martix = self.get_first_order(direct=direct)
        # get_sdp = get_adj_sdp(first_order_dep_adj_matrix)
        first_order_dep_adj_matrix = np.array(first_order_dep_adj_matrix)       
        G = nx.from_numpy_matrix(first_order_dep_adj_matrix) 
        for start_index in start_range+end_range:
            for next_id in range(len(first_order_dep_type_martix[start_index])):
                if first_order_dep_type_martix[start_index][next_id] in ["none"]:
                    continue    ## 该条边不存在关系
                dep_path_adj_matrix[start_index][next_id] = first_order_dep_adj_matrix[start_index][next_id]
                dep_path_type_matrix[start_index][next_id] = first_order_dep_type_martix[start_index][next_id]
                dep_path_adj_matrix[next_id][start_index] = first_order_dep_adj_matrix[next_id][start_index]
                dep_path_type_matrix[next_id][start_index] = first_order_dep_type_martix[next_id][start_index]
        for start_index in start_range: ## 每个头尾实体可能有多个字组成
            # _, sdps = get_sdp.dijkstra(start_index)
            sdps_tuple = list(nx.single_source_dijkstra_path(G,source=start_index).items())
            sdps_tuple = sorted(sdps_tuple, key = lambda k :k[0])
            idx, sdps = zip(*sdps_tuple)
            for end_index in end_range:                
                dep_path_indexs = sdps[end_index]
                # _, dep_path_indexs = self.search_dep_path(start_index, end_index, first_order_dep_type_martix, [start_index])
                for left_index, right_index in zip(dep_path_indexs[:-1], dep_path_indexs[1:]):  ##组合最短依存路径，并遍历
                    dep_path_adj_matrix[left_index][right_index] = 1
                    dep_path_type_matrix[left_index][right_index] = first_order_dep_type_martix[left_index][right_index]
                    dep_path_adj_matrix[right_index][left_index] = 1
                    dep_path_type_matrix[right_index][left_index] = first_order_dep_type_martix[right_index][left_index]                    
        return dep_path_adj_matrix, dep_path_type_matrix

    def get_distance_graph(self, args, direct=True):  ## 使用dijkstra的方法获取SDP
        '''距离加权图 distance'''    
        dep_path_adj_matrix, dep_path_type_matrix = self.get_init_dep_matrix()
        first_order_dep_adj_matrix, first_order_dep_type_martix = self.get_first_order(direct=direct)
        # get_sdp = get_adj_sdp(first_order_dep_adj_matrix)
        first_order_dep_adj_matrix = np.array(first_order_dep_adj_matrix)
        G = nx.from_numpy_matrix(first_order_dep_adj_matrix)
        if args.max_word_len < len(first_order_dep_adj_matrix): ## 选择一个短的，加速计算
            index_range = args.max_word_len
        else:
            index_range = len(first_order_dep_adj_matrix)
        for start_index in range(index_range): ## 所有节点
            if sum(first_order_dep_adj_matrix[start_index]) == 1:    ## 当前节点与其他节点无关系
                dep_path_adj_matrix[start_index][start_index] = 1    ##  对角线值置为1，表示当前节点
                continue
            # distances, sdps = get_sdp.dijkstra(start_index)
            dist = list(nx.single_source_dijkstra_path_length(G,source=start_index).items())
            dist_tuple = sorted(dist, key = lambda dist :dist[0])
            idx, distances = zip(*dist_tuple)
            distances = np.array(distances)
            distances[start_index] = 1 ## ## 自身节点最短距离为0，但要加入自环所以加1,否则自身节点被放大太多
            distances = 1/np.exp(distances-1)    ## w = 1/e^(d-1)
            dep_path_adj_matrix[start_index] = distances  ## 此为距离加权矩阵         
        return dep_path_adj_matrix, first_order_dep_type_martix    ## 使用距离加权矩阵时，不在此处获取依存类型矩阵