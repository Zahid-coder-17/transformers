import torch
import torch.nn as nn
import torch.nn.functional as F

class KV_cache(nn.Module):
    def __init__(self):
        super().__init__()
        self.keys = None
        self.values = None
        
    def update(self, K, V):
        if self.keys is None:
            self.keys = K
            self.values = V
        else:
            self.keys = torch.cat([self.keys, K], dim=2)
            self.values = torch.cat([self.values, V], dim=2)
        return self.keys, self.values
    
    def reset(self):
        self.keys = None
        self.values = None