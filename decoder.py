import torch 
import torch.nn as nn

from encoder import *


class DecoderBlock(nn.Module):

    def __init__(self, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, feed_forward: FeedForward, dropout: float):
        super().__init__()

        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connection = nn.Module([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):

        # self attention for decoder
        x = self.residual_connection[0](x, lambda x:self.self_attention(x,x,x,tgt_mask))
        x = self.residual_connection[1](x, lambda x:self.self_attention(x, encoder_output, x, src_mask))
        x = self.residual_connection[2](x, self.feed_forward)

        return x

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return self.norm(x)

class PredictionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size: int):
        super().__init__()

        self.linear = nn.linear(d_model, vocab_size)
    
    def forward(self, x):
        '''
        x: batch_size, seq_len, d_model
        '''
        return torch.log_softmax(self.linear(x), dim = -1)
