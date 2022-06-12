import torch
from torch import nn

class Add(nn.Module):
    def __init__(self):
        super().__init__()
        self.input1_fake_quant_scale = None
        self.input2_fake_quant_scale = None

    def forward(self, input1, input2):
        pass
