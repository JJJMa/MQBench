import torch.nn.qat.modules as nnqat

# from mqbench.quantization.default_bias_fake_quant import bias_fake_quantizer

class Conv2d(nnqat.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', qconfig=None, device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, qconfig=qconfig) 
        if self.qconfig.bias() is None:
            self.bias_fake_quant = nn.Identity()
        else:
            self.bias_fake_quant = self.qconfig.bias()   
            self.input_fake_quant = None

    def forward(self, input):
        if self.qconfig.bias() and self.input_fake_quant:
            self.bias_fake_quant.scale = self.weight_fake_quant.scale.data * self.input_fake_quant.scale.data
            self.bias_fake_quant.disable_observer()
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias_fake_quant(self.bias)) 
