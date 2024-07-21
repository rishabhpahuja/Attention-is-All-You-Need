import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.trainers import WordLevelTrainer
from tokenizers.trainers import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_all_sentences(ds, lang):
    for item in ds:
        yield  item['translation'][lang]

def get_or_build_tokenzier(config, ds, lang):

    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):

        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace() # Divide sentences by white spaces
        # the word needs to appear atleast twice to appear in vocabulary
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOD]"], min_frequency=2) 
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    
    else:
        tokenzier = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def get_ds(config):

    ds_raw = load_dataset('opus_books',f'{config["lang-src"]-config["lang-tgt"]}')

    # Build Tokenizer
    tokenizer_src  = get_or_build_tokenzier(config, ds_raw, config['lang-src'])
    tokenizer_tgt  = get_or_build_tokenzier(config, ds_raw, config['lang-tgt'])

    # Keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = torch.utils.data.random_split(ds_raw, [train_ds_size, val_ds_size])

      
