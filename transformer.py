import torch
import torch.nn as nn
import torch.nn.functional as F
from normalization.rms_norm import RMSNorm
from normalization.layernorm import LayerNorm
from feedforward.swiglu import SwiGlu
from feedforward.geglu import GEGLU
from attention.mha import MultiHeadAttention
from attention.mqa import MultiQueryAttention
from attention.gqa import GroupedQueryAttention
from attention.self_attention import SelfAttention

class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,hidden_dim,attention,position_encoding,norm_type,ffn_type,num_kv_heads):
        super().__init__()
        
        if norm_type == "rms":
            self.norm1 = RMSNorm(d_model)
        elif norm_type == "layer":
            self.norm1 = LayerNorm(d_model)
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")
            
        if attention == "mha":
            self.attn = MultiHeadAttention(d_model,num_heads,position_encoding)
        elif attention == "mqa":
            self.attn = MultiQueryAttention(d_model,num_heads,position_encoding)
        elif attention == "gqa":
            if num_kv_heads is None:
                raise ValueError("num_kv_heads must be specified for GroupedQueryAttention")
            self.attn = GroupedQueryAttention(d_model,num_heads,num_kv_heads, position_encoding)
        elif attention == "self":
            self.attn = SelfAttention(d_model)
        else:
            raise ValueError(f"Unsupported attention type: {attention}")
            
        if norm_type == "rms":
            self.norm2 = RMSNorm(d_model)
        elif norm_type == "layer":
            self.norm2 = LayerNorm(d_model)
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")
              
        
        if ffn_type == "swiglu":
            self.ffn = SwiGlu(d_model,hidden_dim)
        elif ffn_type == "geglu":
            self.ffn = GEGLU(d_model,hidden_dim)
        else:
            raise ValueError(f"Unsupported feed-forward type: {ffn_type}")
    
    
    def forward(self,x):
        attn_out = self.attn(self.norm1(x))
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x 