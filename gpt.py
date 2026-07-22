import torch.nn as nn
import torch
from transformer import TransformerBlock
from normalization.rms_norm import RMSNorm
from normalization.layernorm import LayerNorm
from position.sinusodal import SinusoidalPositionalEncoding
from position.learnedpe import LearnedPositionalEncoding


class GPT(nn.Module):
    def __init__(self,vocab_size,d_model,num_heads,hidden_dim,num_layers,attention_type,normalization_type,feedforward_type,position_encoding,max_seq_len=4096,num_kv_heads=None):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size,d_model)
        
        if position_encoding == "absolute":
            self.position_embedding = nn.Embedding(max_seq_len,d_model)
        elif position_encoding == "sinusoidal":
            self.position_embedding = SinusoidalPositionalEncoding(d_model,max_seq_len)
        elif position_encoding == "learned":
            self.position_embedding = LearnedPositionalEncoding(d_model,max_seq_len)
        elif position_encoding in ["rope", "alibi"]:
            self.position_embedding = None 
        else:
            raise ValueError(f"Unsupported position encoding type: {position_encoding}")

        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model,num_heads=num_heads,hidden_dim=hidden_dim,attention=attention_type,position_encoding=position_encoding,
                norm_type=normalization_type,ffn_type=feedforward_type,num_kv_heads=num_kv_heads)
            for _ in range(num_layers)
        ])
        
        if normalization_type == "rms":
            self.norm = RMSNorm(d_model)
        elif normalization_type == "layer":
            self.norm = LayerNorm(d_model)
        else:
            raise ValueError(f"Unsupported normalization type: {normalization_type}")
        
        self.lm_head = nn.Linear(d_model,vocab_size,bias= False)
        
    def forward(self, input_ids):

        x = self.embedding(input_ids)

        if self.position_embedding is not None:
            if isinstance(self.position_embedding, nn.Embedding):

                batch_size, seq_len = input_ids.shape
                positions = torch.arange(
                seq_len,
                device=input_ids.device
                ).unsqueeze(0)
                positions = positions.expand(batch_size, seq_len)
                x = x + self.position_embedding(positions)
            else:
                x = self.position_embedding(x)
                
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits