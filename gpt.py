import torch
import torch.nn as nn
from transformer import TransformerBlock
from rms_norm import RMSNorm

class GPT(nn.Module):
    def __init__(self,vocab_size,d_model,num_heads,hidden_dim,num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model,num_heads,hidden_dim)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model,vocab_size,bias= False)
        
    def forward(self,x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits