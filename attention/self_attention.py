import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.d_model = d_model
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
    
    def forward(self,x):
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        scores = Q @ K.transpose(-2,-1)
        scores = scores / math.sqrt(self.d_model)
        attention = F.softmax(scores,dim=-1)
        output = attention @ V
        
        return output
    
