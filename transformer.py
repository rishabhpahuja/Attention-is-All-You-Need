import torch
import torch.nn as nn

from encoder import *
from decoder import *

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder,decoder: Decoder, src_embed: InputEmbedding, 
                 tgt_embed: InputEmbedding, src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding, prediction_layer: PredictionLayer):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.prediction_layer = prediction_layer

    def encode(self, src, src_mask):

        x = self.src_embed(src)
        x = self.src_pos(x)

        return self.encoder(x, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):

        x = self.tgt_embed(tgt)
        x = self.tgt_pos(x)

        return self.decoder(x, encoder_output, src_mask, tgt_mask)
    
    def predict(self, x):

        return self.prediction_layer(x)

def build_transformer(src_vocab: int, tgt_vocab: int, src_seq_len: int, 
                 tgt_seq_len: int, d_model: int = 512, N:int = 6, n_heads: int = 8,
                  dropout: float = 0.1, d_ff = 2048):
        
        src_embed = InputEmbedding(d_model, src_vocab)
        tgt_embed = InputEmbedding(d_model, tgt_vocab)

        src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
        tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

        # Create Encoder Blocks
        encoder_blocks = []

        for i in range(N):
            self_attention_block = MultiHeadAttention(d_model, n_heads, dropout)
            feed_forward = FeedForward(d_model, d_ff, dropout )
            encoder_block = EncoderBlock(self_attention_block, feed_forward, dropout)
            encoder_blocks.append(encoder_block)

        decoder_blocks = []

        for i in range(N):
            decoder_self_attention_block = MultiHeadAttention(d_model, n_heads, dropout)
            decoder_cross_attention_block = MultiHeadAttention(d_model, n_heads, dropout)
            decoder_feed_forward = FeedForward(d_model, d_ff, dropout )
            decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward, dropout)
            decoder_blocks.append(decoder_block)
        
        encoder = Encoder(nn.ModuleList(encoder_blocks))
        decoder = Decoder(nn.ModuleList(decoder_blocks))

        prediction_layer = PredictionLayer(d_model, tgt_vocab)

        transformer = Transformer(encoder, decoder, src_embed, tgt_embed,
                                       src_pos, tgt_pos, prediction_layer 
                                       )
        
        return transformer
