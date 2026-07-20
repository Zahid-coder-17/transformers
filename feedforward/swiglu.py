import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGlu(nn.Module):
    def __init__(self,d_model,hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(d_model,hidden_dim)
        self.up_proj = nn.Linear(d_model,hidden_dim)
        self.down_proj = nn.Linear(hidden_dim,d_model)
        
    def forward(self,x):
        gate = self.gate_proj(x)
        value = self.up_proj(x)
        out = F.silu(gate) * value
        out = self.down_proj(out)
        return out 