import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size:int):
        super().__init_()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):

        return self.embedding(x)*(self.d_model**0.5)
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        # data structure to store the encodings
        pe = torch.zeros(seq_len, d_model)
        # Numerator of encoding
        pos = torch.arange(seq_len, dtype = torch.float).unsqueeze(0)
        # Denominotr of encoding
        denom = torch.exp(torch.arange(0,d_model,2)*(-math.log(10000))/d_model)
        # Even position encoding
        pe[:,0::2] = torch.sin(pos*denom)
        # Odd position encoding
        pe[:,1::2] = torch.cos(pos*denom)

        self.pe = pe.unsqueeze(0) # To add batch dimension

        self.register_buffer('pe', pe) # This tensor will be saved along with the model

    
    def forward(self,x):

        x = x + (self.pe[:, x.shape[1],:]).requires_grad(False)

        return self.dropout(x)




