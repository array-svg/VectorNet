from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import MessagePassing, max_pool
import numpy as np
import pandas as pd
from utils.viz_utils import show_predict_result
import matplotlib.pyplot as plt
import numpy as np
import pdb
import os
import d2l
def SequenceMask(X, X_len,value=-1e6):
    maxlen = X.size(1)
    X_len = X_len.to(X.device)
    #print(X.size(),torch.arange((maxlen),dtype=torch.float)[None, :],'\n',X_len[:, None] )
    mask = torch.arange((maxlen), dtype=torch.float, device=X.device)
    mask = mask[None, :] < X_len[:, None]
    #print(mask)
    X[~mask]=value
    return X

def masked_softmax(X, valid_lens):
    """
    masked softmax for attention scores
    args:
        X: 3-D tensor, valid_len: 1-D or 2-D tensor
    """
    # if valid_len is None:
    #     return nn.functional.softmax(X, dim=-1)
    # else:
    #     shape = X.shape
    #     if valid_len.dim() == 1:
    #         valid_len = torch.repeat_interleave(
    #             valid_len, repeats=shape[1], dim=0)
    #     else:
    #         valid_len = valid_len.reshape(-1)
    #     # Fill masked elements with a large negative, whose exp is 0
    #     X = X.reshape(-1, shape[-1])
        
    #     # for count, row in enumerate(X):
    #         # row[int(valid_len[count]):] = -1e6
        
    #     # 最后一个问题，不知道如何解决
    #     X_copy = X.clone()
    #     for count, row in enumerate(X_copy):
    #     # 修改副本而不是原始的视图
    #         row[int(valid_len[count]):] = -1e6
        
    #     return nn.functional.softmax(X.reshape(shape), dim=-1)
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后⼀轴上被掩蔽的元素使⽤⼀个⾮常⼤的负值替换，从⽽其softmax输出为0
        X = SequenceMask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class SelfAttentionLayer(nn.Module):
    """
    Self-attention layer. no scale_factor d_k
    """

    def __init__(self, in_channels, global_graph_width, need_scale=False):
        super(SelfAttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.q_lin = nn.Linear(in_channels, global_graph_width)
        self.k_lin = nn.Linear(in_channels, global_graph_width)
        self.v_lin = nn.Linear(in_channels, global_graph_width)
        self.scale_factor_d = 1 + \
            int(np.sqrt(self.in_channels)) if need_scale else 1

    def forward(self, x, valid_len):
        # print(x.shape)
        # print(self.q_lin)
        print("a line 55")
        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)
        print("a line 59")
        scores = torch.bmm(query, key.transpose(1, 2))
        print("a line 61")
        attention_weights = masked_softmax(scores, valid_len)
        print("a line 63")
        return torch.bmm(attention_weights, value)
