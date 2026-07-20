import torch
import torch.nn as nn
import torch.nn.functional as F
from rms_norm import RMSNorm
from multi_head_attention import MultiHeadAttention
from swiglu import SwiGlu

class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,hidden_dim):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model,num_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGlu(d_model,hidden_dim)
        
    def forward(self,x):
        attn_out = self.attn(self.norm1(x))
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x 