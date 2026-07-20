import torch
import torch.nn as nn
import torch.nn.functional as F
from kv_cache import KV_cache
import math

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)
        self.cache = KV_cache()
        
    def forward(self,x):
        B,T,D = x.shape
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        Q = Q.view(B,T,self.num_heads,self.head_dim)
        K = K.view(B,T,self.num_heads,self.head_dim)
        V = V.view(B,T,self.num_heads,self.head_dim)
        
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)
        
        scores = Q @ K.transpose(-2,-1)
        scores = scores/math.sqrt(self.head_dim)
        
        attention = F.softmax(scores,dim=-1)
        
        output = attention @ V 
        output = output.transpose(1,2)
        output = output.contiguous().view(B,T,D)
        output = self.w_o(output)
        
        return output