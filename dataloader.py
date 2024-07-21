import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang,
                 tgt_lang, seq_len):
        
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype = torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype = torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype = torch.int64)
    
    def __len__(self):
        return len(self.ds)
    
    def __getitm__(self, index):

        src_tgt_pair = self.ds[index]
        src_text = src_tgt_pair['translation'][self.src_lang]
        tgt_text = src_tgt_pair['translation'][self.tgt_lang]

        # Converting each word of a sentence to an id
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids 
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding = self.seq_len - len(enc_input_tokens) - 2 # -2 because <EOS> and <SOS> shall be added
        dec_num_padding = self.seq_len - len(dec_input_tokens) - 1 # only <EOS> is added to output tokens

        if enc_num_padding < 0 or dec_num_padding < 0:
            raise ValueError('Sentence is too long')
        
        # Add SOS,ID, SOS and PAD
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*enc_num_padding, dtype = torch.int64)
            ]
        )

        # Add SOS, ID and PAD
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token]*dec_num_padding)
            ]
        )

        # Add EOS to the label
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*dec_num_padding, dtype = torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {"encoder_input":encoder_input,
                "decoder_input": decoder_input,
                "encoder_mask": (encoder_input !=self.pad_token).unsqueeze(0).unsqueeze(0).int(), # shape: [1,1,seq_len]
                "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
                "src_text": src_text,
                "tgt_text": tgt_text

                }

def causal_mask(size):

    mask =  torch.triu(torch.ones(1, size, size),diagonal = 1).type(torch.int)
    return mask == 0