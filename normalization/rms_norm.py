import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self,d_model,eps=1e-8):
        super().__init__()
        self.d_model = d_model 
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self,x):
        rms = torch.sqrt(torch.mean(x ** 2 ,dim = -1 ,keepdim=True)+self.eps)
        x = x/rms
        x = x*self.weight
        return x 