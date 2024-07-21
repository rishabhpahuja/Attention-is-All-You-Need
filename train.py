import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_or_build_tokenzier(config, ds, lang):

    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):

        tokenizer = Tokenizer(WordLevelTrainer(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace() # Divide sentences by white spaces

      
