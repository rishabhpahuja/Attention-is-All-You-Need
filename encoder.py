import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size:int):
        super().__init__()

        '''
        d_model: dimension of embedding
        vocab_size: 
        
        '''
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):

        return self.embedding(x)*(self.d_model**0.5)
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model:int, seq_len:int, dropout:float):

        '''
        d_model: dimension of embedding
        seq_len: length of the sentence
        dropout: probability of dropout
        
        '''
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        # data structure to store the encodings
        pe = torch.zeros(seq_len, d_model)
        # Numerator of encoding
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Denominotr of encoding
        denom = torch.exp(torch.arange(0,d_model,2)*(-math.log(10000))/d_model)
        # Even position encoding
        pe[:, 0::2] = torch.sin(pos * denom)
        # Odd position encoding
        pe[:,1::2] = torch.cos(pos*denom)

        self.pe = pe.unsqueeze(0) # To add batch dimension

        # self.register_buffer('pe', pe)
        # if not hasattr(self, 'pe'):
        #     self.register_buffer('pe', pe)
        # else:
        #     self.pe = pe.unsqueeze(0)

    
    def forward(self,x):
        '''
        x: [batch_size, seq_len, d_model]
        '''
        
        # import ipdb; ipdb.set_trace()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x + (self.pe[:, :x.shape[1],:]).to(device).requires_grad_(False)

        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps:float = 10**-6):
        super().__init__()

        self. eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)

        return self.alpha*(x-mean)/(std+self.eps) + self.bias

class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: int):
        super().__init__()

        self.feed = nn.Sequential(
            nn.Linear(d_model,  d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
                          )

    def forward(self, x):

        return self.feed(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: int):

        '''
        d_model: dimension of embedding
        n_heads: number of attention heads
        dropout: probability of dropout

        '''
        super().__init__()

        assert d_model%n_heads == 0, "d_model should be competely divisible by h"

        self.n_heads = n_heads
        self.d_k = d_model//n_heads # The portion of each embedding
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def attention(self, query, key, value, mask, dropout: nn.Dropout):

        '''
        query, key, value : batch_size, n_heads, seq_len, d_k
        '''

        d_k = query.shape[-1]

        scores = (query @ key.transpose(-2, -1))/d_k**0.5 # [batch_size, n_heads, seq_len, seq_len]

        if mask is not None:
            scores.masked_fill(mask==0, -1e9)
        
        scores = scores.softmax(dim = -1) # [batch_size, n_heads, seq_len, seq_len]

        if dropout is not None:
            scores = dropout(scores) 

        return  scores@value, scores
            
    def forward(self, q, k , v, mask):

        query = self.query(q)
        key = self.key(k)
        value = self.value(v)

        # (batch_size, sequence_len, d_model) -> (batch_size, sequence_len, n_heads, d_k) -> (batch_size, n_heads, sequence_len , d_k)
        query = query.view(query.shape[0],query.shape[1], self.n_heads, self.d_k).transpose(1,2)
        key = key.view(query.shape[0],key.shape[1], self.n_heads, self.d_k).transpose(1,2)
        value = value.view(query.shape[0],value.shape[1], self.n_heads, self.d_k).transpose(1,2)

        x, self.scores = self.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0],-1, self.n_heads*self.d_k) #  [batch_size, seq_len, d_model]

        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self, x, sublayer):

        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, attention_block: MultiHeadAttention, feed_forward: FeedForward, dropout:float):
        super().__init__()

        self.self_attention = attention_block
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):

        x = self.residual_connection[0](x , lambda x:self.self_attention(x,x,x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward)

        return x

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)