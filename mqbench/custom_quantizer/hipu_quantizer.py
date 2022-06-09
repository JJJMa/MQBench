import torch
import torch.nn as nn
import torch.nn.qat as nnqat
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.qat as nniqat
from torch.fx import GraphModule
from torch.nn.intrinsic import (
    _FusedModule
)
from torch.quantization import (
    propagate_qconfig_,
    swap_module
)
from torch.quantization.quantize_fx import (
    _fuse_fx
)

from mqbench.utils.registry import register_model_quantizer
from mqbench.prepare_by_platform import BackendType
from mqbench.custom_quantizer import TotalINTQuantizer
from mqbench.utils.logger import logger
import mqbench.nn.intrinsic.qat as qnniqat
import mqbench.nn.qat as qnnqat

@register_model_quantizer(BackendType.Hipu)
class HipuQuantizer(TotalINTQuantizer):
    """
    The quantization style used in this code is Power-Of-2, Symmetric, Per-Tensor Quantization for both Weights and Activations. \
    There is also an option to use Per-Channel Weight Quantization for Depthwise Convolution Layers.
    https://github.com/TexasInstruments/edgeai-torchvision/blob/master/docs/pixel2pixel/Quantization.md
    """

    def __init__(self, extra_quantizer_dict, extra_fuse_dict):
        super().__init__(extra_quantizer_dict, extra_fuse_dict)
        self.additional_qat_module_mapping = {
            nn.Conv2d: qnnqat.Conv2d,
            nn.Linear: qnnqat.Linear,
            # Intrinsic modules:
            nni.ConvBn2d: qnniqat.ConvBn2d,
            nni.ConvReLU2d: qnniqat.ConvReLU2d,
            nni.ConvBnReLU2d: qnniqat.ConvBnReLU2d,
        }
        self.additional_module_type = (
            qnnqat.Conv2d,
            qnnqat.Linear,
            qnniqat.ConvBn2d,
            qnniqat.ConvReLU2d,
            qnniqat.ConvBnReLU2d,
        )

    @property
    def module_contain_weight_bias(self) -> tuple:
        return self.additional_module_type

    def prepare(self, model: GraphModule, qconfig):
        model = _fuse_fx(model, self.extra_fuse_dict)
        model = self._weight_quant(model, qconfig)
        model = self._insert_fake_quantize_for_act_quant(model, qconfig)

        # TODO
        # self._binding_act_quant_to_module(model)
        return model
    
    def _binding_act_quant_to_module(self, model: GraphModule):
        nodes = list(model.graph.nodes)
        modules = dict(model.named_modules())
        for node in nodes:
            if (node.op == "call_module" and isinstance(modules[node.target], self.module_contain_weight_bias)):
                modules[node.target].input_fake_quant = modules[node.args[0].target]
                logger.info(f"Binding act quant node: {node.args[0].target} to {node.target}.")
        return
