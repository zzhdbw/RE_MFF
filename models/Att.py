## time :22-03-09_15.26.34
## author: zjp
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention

class Att_self(nn.Module):
    def __init__(self, attention_size, max_len, hidden_size):
        '''自注意力 注意力尺寸 句子长度 输入的隐藏维度'''
        super(Att_self, self).__init__()
        '''
        W: (hidden_size, attention_size)
        b: (attention_size, 1)
        u: (attention_size, 1)
        '''
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.max_len = max_len

        self.W = nn.Parameter(torch.FloatTensor(self.hidden_size,self.attention_size))
        # self.W = nn.Linear(self.hidden_size,self.attention_size)
        self.b = nn.Parameter(torch.FloatTensor(self.max_len, 1))
        self.u = nn.Parameter(torch.FloatTensor(self.attention_size, 1))
        ## 指定初始化权重的方式
        nn.init.xavier_normal_(self.W)
        nn.init.constant_(self.b, val=0)
        nn.init.xavier_normal_(self.u)

        self.Tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, attention_mask=None):
        '''
        query: (batch_size, maxlen, hidden_size)
        et: (batch_size, maxlen, attention_size)
        at: (batch_size, maxlen)
        ot: (batch_size, maxlen, hidden_size)
        output: (batch_size, hidden_size)
        '''
        et = self.Tanh(torch.matmul(query, self.W) + self.b)

        # print("et:",et.shape)
        at = self.softmax(torch.matmul(et, self.u).squeeze(-1))
        # print("at:",at.shape)   ## [1, 15]

        ot = torch.mul(at.unsqueeze(-1) , query)    ## [1, 160, 256]

        # print("ot:",ot.shape)
        output = torch.sum(ot, dim=1)       ## [1, 256]
        return output
