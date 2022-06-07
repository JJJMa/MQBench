import torch
from torch.fx import GraphModule
from torch.nn.intrinsic import (
    _FusedModule
)
from torch.quantization import (
    propagate_qconfig_,
    swap_module
)

from mqbench.utils.registry import register_model_quantizer
from mqbench.prepare_by_platform import BackendType
from mqbench.custom_quantizer import TotalINTQuantizer
from mqbench.utils.logger import logger


@register_model_quantizer(BackendType.Ti)
class TiQuantizer(TotalINTQuantizer):
    """
    The quantization style used in this code is Power-Of-2, Symmetric, Per-Tensor Quantization for both Weights and Activations. \
    There is also an option to use Per-Channel Weight Quantization for Depthwise Convolution Layers.
    https://github.com/TexasInstruments/edgeai-torchvision/blob/master/docs/pixel2pixel/Quantization.md
    """

    def __init__(self, extra_quantizer_dict, extra_fuse_dict):
        super().__init__(extra_quantizer_dict, extra_fuse_dict)

    def _convert(self, module, mapping=None, inplace=False, scope=''):
        if mapping is None:
            mapping = get_default_static_quant_module_mappings()

        if not inplace:
            module = copy.deepcopy(module)
        reassign = {}
        for name, mod in module.named_children():
            # fused modules are swapped as one unit
            new_scope = "{}.{}".format(scope, name) if scope != '' else name
            if new_scope in self.exclude_module_name:
                logger.info("Skip quant layer: " + new_scope)
                continue
            if not isinstance(mod, _FusedModule):
                self._convert(mod, mapping, True, new_scope)
            reassign[name] = swap_module(mod, mapping, {})

            if self._contain_dwconv(mod):
                if hasattr(reassign[name], "weight_fake_quant") and reassign[name].weight_fake_quant.ch_axis == -1:
                    reassign[name].weight_fake_quant.ch_axis = 0
                    reassign[name].weight_fake_quant.activation_post_process.ch_axis = 0
                    reassign[name].weight_fake_quant.qscheme = torch.per_channel_symmetric
                    reassign[name].weight_fake_quant.is_per_channel = True
                    logger.info("Switch DWconv fake quant to per-channel: " + new_scope)
        for key, value in reassign.items():
            module._modules[key] = value

        return module

    def _contain_dwconv(self, mod):
        if isinstance(mod, (torch.nn.Conv2d,)) and mod.groups == mod.in_channels:
            return True
        elif any(self._contain_dwconv(m) for n, m in mod.named_children()):
            return True
        else:
            return False
