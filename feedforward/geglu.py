import torch
import torch.nn as nn
import torch.nn.functional as F


class GEGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim)
        self.w2 = nn.Linear(d_model, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        a = self.w1(x)
        b = self.w2(x)
        gate = F.gelu(b)
        x = a * gate
        return self.w3(x)

