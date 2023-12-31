# -- coding: utf-8 --**
# @author zjp
# @time 22-05-11_09.59.47
## 该gcn为简化版，将(text*weight*adj)/度，并未进行，行列归一化即没有左乘右乘度矩阵
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    '''base gcn'''
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
    
    def forward(self, text, adj):
        # text.shape [1, 80, 256]
        # self.weight.shape[256, 256]
        # adj [1, 80, 80]
        # denom [1, 80, 1]
        hidden = torch.matmul(text.float(), self.weight.float())    ## [1, 80, 256]
        denom = torch.sum(adj, dim=2, keepdim=True) + self.b ## 求每个节点的度 为何加1 防止下面除denom 出错,当使用重整化参数时不加零
        output = torch.matmul(adj.float(), hidden)/denom    ## [1, 80, 256]  ##
        # output = torch.matmul(adj/denom , hidden)   ## [1, 80, 256]  
        if self.bias is not None:
            output = output + self.bias
        # output = self.fc(output)
        return F.relu(output.type_as(text))

