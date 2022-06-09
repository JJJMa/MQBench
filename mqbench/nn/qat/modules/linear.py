import torch.nn.qat.modules as nnqat
import torch.nn.functional as F


class Linear(nnqat.Linear):
    def __init__(self, in_features, out_features, bias=True, qconfig=None, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, qconfig=qconfig)
        if self.qconfig.bias() is None:
            self.bias_fake_quant = nn.Identity()
        else:
            self.bias_fake_quant = self.qconfig.bias()   
            self.input_fake_quant = None

    def forward(self, input):
        if self.qconfig.bias() and self.input_fake_quant:
            self.bias_fake_quant.scale = self.weight_fake_quant.scale.data * self.input_fake_quant.scale.data
            self.bias_fake_quant.disable_observer()
        return F.linear(input, self.weight_fake_quant(self.weight), self.bias_fake_quant(self.bias)) 
