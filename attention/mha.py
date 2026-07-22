import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.kv_cache import KV_cache
import math
from position.rope import apply_rotary_pos_emb, get_rope_sin_cos
from position.alibi import build_alibi_bias, apply_alibi_bias
from utils.mask import build_causal_mask, apply_causal_mask

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads,position_encoding=None,use_mask=True,use_kv_cache=False):
        super().__init__()
        
        self.position_encoding = position_encoding
        self.use_mask = use_mask
        self.use_kv_cache = use_kv_cache

        if self.use_kv_cache:
            self.kv_cache = KV_cache()
    
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
        
        if self.position_encoding == "rope":
            sin, cos = get_rope_sin_cos(T, self.head_dim, Q.device)
            Q = apply_rotary_pos_emb(Q, sin, cos)
            K = apply_rotary_pos_emb(K, sin, cos)
        
        scores = Q @ K.transpose(-2,-1)
        scores = scores/math.sqrt(self.head_dim)
        
        if self.position_encoding == "alibi":
            bias = build_alibi_bias(T, self.num_heads).to(scores.device)
            scores = apply_alibi_bias(scores, bias)
            
        if self.use_mask:
            causal_mask = build_causal_mask(T).to(scores.device)
            scores = apply_causal_mask(scores, causal_mask)
        
        attention = F.softmax(scores,dim=-1)
        
        output = attention @ V 
        output = output.transpose(1,2)
        output = output.contiguous().view(B,T,D)
        output = self.w_o(output)
        
        return output
    
    