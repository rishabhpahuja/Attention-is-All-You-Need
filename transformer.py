import torch
import torch.nn as nn

from encoder import *
from decoder import *

class Tranformer(nn.Module):
    def __init__(self, encoder: Encoder,decoder: Decoder):
        super().__init__()
