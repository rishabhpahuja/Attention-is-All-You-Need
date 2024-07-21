import torch
import torch.nn as nn

from encoder import *
from decoder import *

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder,decoder: Decoder, src_embed: InputEmbedding, 
                 tgt_embed: InputEmbedding, src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding, pridiction_layer: PredictionLayer):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.prediction_layer = pridiction_layer

    def encode(self, src, src_mask):

        x = self.src_embed(src)
        x = self.src_pos(x)

        return self.encoder(x, src_mask)

    def decode(self, tgt, tgt_mask):

        x = self.tgt_embed(tgt)
        x = self.tgt_pos(x)

        return self.decoder(x, tgt_mask)
    
    def predict(self, x):

        return self.prediction_layer(x)

class TransformerModel(nn.MOdule):
    def __init__(self, src_vocab: int, tgt_vocab: int, src_seq_len: int, 
                 tgt_seq_len: int, d_model: int = 512, N:int = 6, n_heads: int = 8,
                  dropout: float = 0.1, d_ff = 2048):
        
        self.src_embed = InputEmbedding(d_model, src_vocab)
        self.tgt_embed = InputEmbedding(d_model, tgt_vocab)

        self.src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
        self.tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

        # Create Encoder Blocks
        self.encoder_blocks = []

        for i in range(N):
            self_attention_block = MultiHeadAttention(d_model, n_heads, dropout)
            feed_forward = FeedForward(d_model, d_ff, dropout )
            encoder_block = EncoderBlock(self_attention_block, feed_forward, dropout)
            self.encoder_blocks.append(encoder_block)

        self.decoder_blocks = []

        for i in range(N):
            decoder_self_attention_block = MultiHeadAttention(d_model, n_heads, dropout)
            decoder_cross_attention_block = MultiHeadAttention(d_model, n_heads, dropout)
            decoder_feed_forward = FeedForward(d_model, d_ff, dropout )
            decoder_block = EncoderBlock(decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward, dropout)
            self.decoder_blocks.append(decoder_block)
        
        self.encoder = Encoder(nn.ModuleList(self.encoder_blocks))
        self.decoder = Decoder(nn.ModuleList(self.decoder_blocks))

        self.prediction_layer = PredictionLayer(d_model, tgt_vocab)

        self.Transformer = Transformer(self.encoder, self.decoder, self.src_embed, self.tgt_embed,
                                       self.src_pos, self.tgt_pos 
                                       )
    
    def forward(self, x):

        return self.Transformer(x)


