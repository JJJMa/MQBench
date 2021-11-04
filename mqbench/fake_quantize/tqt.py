from functools import partial

import torch
from torch.nn.parameter import Parameter

from mqbench.fake_quantize.quantize_base import QuantizeBase
from mqbench.utils import is_symmetric_quant


class TqtFakeQuantize(QuantizeBase):
    def __init__(self, observer, scale=1., zero_point=0., use_grad_scaling=True, **observer_kwargs):
        super(TqtFakeQuantize, self).__init__(observer, **observer_kwargs)
        self.scale = Parameter(torch.tensor([scale]))
        self.zero_point = Parameter(torch.tensor([zero_point]))
        self.register_buffer('eps', torch.tensor([torch.finfo(torch.float32).eps]))

        class PerChannelLoadHook:
            def __init__(self, module):
                self.hook = module._register_load_state_dict_pre_hook(partial(self.hook_fn, module=module))

            def hook_fn(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
                        module):
                if module.ch_axis == -1:
                    # no per-channel parameters
                    return
                for module_key, param in module._parameters.items():
                    if module_key not in ["scale", "zero_point"]:
                        continue
                    candidate = prefix + module_key
                    if candidate in state_dict:
                        input_param = state_dict[candidate]
                        if param.shape != input_param.shape:
                            param.data = torch.ones_like(input_param, dtype=param.dtype, device=param.device)

            def close(self):
                self.hook.remove()

        self.load_state_dict_hook = PerChannelLoadHook(self)

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={}, ' \
               'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
               'scale={}, zero_point={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   self.quant_min, self.quant_max,
                   self.dtype, self.qscheme, self.ch_axis, self.scale if self.ch_axis == -1 else 'List',
                   self.zero_point if self.ch_axis == -1 else 'List')

    def forward(self, X):
        # Learnable fake quantize have to zero_point.float() to make it learnable.
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            _scale = _scale.to(self.scale.device)
            _zero_point = _zero_point.to(self.zero_point.device)

            if self.ch_axis != -1:
                self.scale.data = torch.ones_like(_scale)
                self.zero_point.data = torch.zeros_like(_zero_point.float())

            self.scale.data.copy_(_scale)
            self.zero_point.data.copy_(_zero_point.float())
        else:
            self.scale.data.abs_()
            self.scale.data.clamp_(min=self.eps.item())

        if self.fake_quant_enabled[0] == 1:
            assert is_symmetric_quant(self.qscheme)
            "TQT is a symmetric quantization FakeQuantize Op."
            self.zero_point.data.zero_()
            assert self.is_per_channel is False 
            "TQT is a per-tensor quantization FakeQuantize Op."
            X = FakeQuantizeTqtAffine.apply(X, self.scale, self.zero_point, self.quant_min, self.quant_max)
        return X


def _fake_quantize_tqt_affine_training(x, scale, zero_point, quant_min, quant_max):    
    return torch.clamp(scale_floor(x / scale), quant_min, quant_max) * scale 


def scale_floor(t):
    return (torch.floor(t) - t).detach() + t 


class FakeQuantizeTqtAffine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, quant_min, quant_max):
        return _fake_quantize_tqt_affine_training(x, scale, zero_point, quant_min, quant_max)

    @staticmethod
    def symbolic(g, x, scale, zero_point, quant_min, quant_max):
        return g.op("::FakeQuantizeTqtAffine", x, scale, zero_point, quant_min_i=quant_min, quant_max_i=quant_max)
