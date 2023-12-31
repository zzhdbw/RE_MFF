# -- coding: utf-8 --**
# @author zjp
# @time 22-05-14_09.59.47
## 该gcn为简化版，将(text*weight*adj)/度，并未进行，行列归一化即没有左乘右乘度矩阵
## 依存关系类型 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    '''type gcn'''
    def __init__(self, input_dim, output_dim, renorm, bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim        
        self.b = 1.0 if renorm == 0 else 0.0   ## renorm == 0:## 不使用重整化参数
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        # self.fc = nn.Linear(in_features=input_dim, out_features=output_dim, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)   ##将self.bias=None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))   ## kaiming 均匀分布
        if self.bias is not None:
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, text, adj, dep_embed):
        ## text.shape [1, 80, 256] ## self.weight.shape[256, 256]
        ## adj [1, 80, 80]  dep_embed [1, 80, 80, hidden]
        batch_size, max_len, feat_dim = text.shape
        val_us = text.unsqueeze(dim=2)  ## 拓展维度、
        val_us = val_us.repeat(1, 1, max_len, 1) ## [batch_size,maxlen,maxlen, hidden_size] 
        val_sum = val_us + dep_embed  ## 将每个词添加该词与其他词之间的依存类型嵌入向量 ## 为什么不是cat, cat的话dep_bedm的维度就都行了
        hidden = torch.matmul(val_sum.float(), self.weight.float())  ## 添加可训练参数
        adj_us = adj.unsqueeze(dim=-1)
        adj_us = adj_us.repeat(1, 1, 1, feat_dim)   ## [batch_size,maxlen,maxlen, hidden_size] 
        output = hidden.transpose(1,2) * adj_us.float() ## 提取与当前节点词有关系的节点，在下一步将其相加进行聚合
        output = torch.sum(output, dim=2)       ## 聚合每一行的信息,是否应该除以度
        if self.bias is not None:
            output = output + self.bias
        # output = self.fc(output)
        return F.relu(output.type_as(text))

