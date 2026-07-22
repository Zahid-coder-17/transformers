import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from position.rope import apply_rotary_pos_emb, get_rope_sin_cos
from position.alibi import build_alibi_bias, apply_alibi_bias
from utils.mask import build_causal_mask, apply_causal_mask

class SelfAttention(nn.Module):
    def __init__(self,d_model,position_encoding=None,use_mask=True):
        super().__init__()
        self.d_model = d_model
        self.position_encoding = position_encoding
        self.use_mask = use_mask
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
    
    def forward(self,x):
        B, T, D = x.shape
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        if self.position_encoding == "rope":
            sin, cos = get_rope_sin_cos(T, D, Q.device)
            Q = apply_rotary_pos_emb(Q, sin, cos)
            K = apply_rotary_pos_emb(K, sin, cos)
        
        scores = Q @ K.transpose(-2,-1)
        scores = scores / math.sqrt(self.d_model)
        
        if self.position_encoding == "alibi":
            bias = build_alibi_bias(T, 1).to(scores.device)
            scores = apply_alibi_bias(scores, bias)
            
        if self.use_mask:
            causal_mask = build_causal_mask(T).to(scores.device)
            scores = apply_causal_mask(scores, causal_mask)
            
        attention = F.softmax(scores,dim=-1)
        output = attention @ V
        
        return output
    
