import torch.nn as nn
import torch
from transformer import TransformerBlock
from normalization.rms_norm import RMSNorm
from normalization.layernorm import LayerNorm
from position.sinusodal import SinusoidalPositionalEncoding
from position.learnedpe import LearnedPositionalEncoding
from data.download import dataset
from tokenization.character import get_batch,decode,vocab_size
import torch.nn.functional as F

class GPT(nn.Module):
    def __init__(self,vocab_size,d_model,num_heads,hidden_dim,num_layers,attention_type,normalization_type,feedforward_type,position_encoding,max_seq_len=4096,num_kv_heads=None,block_size=1024):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.block_size = block_size

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
        
    def forward(self, input_ids,targets=None):

        x = self.embedding(input_ids)

        if self.position_embedding is not None:
            if isinstance(self.position_embedding, nn.Embedding):

                batch_size, seq_len = input_ids.shape
                positions = (torch.arange(
                seq_len,
                device=input_ids.device
                ).unsqueeze(0).expand(batch_size,seq_len))
                
                x = x + self.position_embedding(positions)
            else:
                x = self.position_embedding(x)
                
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B,T,C = logits.shape
            logits_flat = logits.view(B*T,C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat,targets_flat)
        
        
        return logits,loss
    
    
    @torch.inference_mode()        
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
    

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)
    def forward(self,idx,targets = None):
        logits = self.token_embedding_table(idx)
        loss = None
        if targets is not None:
            B,T,C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits,targets)
            return logits,loss
        return logits, loss

    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx

        
if __name__ == "__main__":
    xb, yb = get_batch("train")
    model = BigramLanguageModel()
    logits, loss = model(xb, yb)
    
    print("Logits shape:", logits.shape)
    print("Loss:", loss)
    
    idx = torch.zeros((1,1),dtype=torch.long)
    generated_text = model.generate(idx,100)
    print(decode(generated_text[0].tolist()))



    