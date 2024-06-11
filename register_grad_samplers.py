from typing import Dict, List

import torch
import torch.nn as nn
# from opacus_nn_model_jp_2 import DPMultiheadAttention
# from imitator.models.nn_model_jp import TransformerDecoderLayer

from opacus.grad_sample.utils import register_grad_sampler

@register_grad_sampler(nn.TransformerDecoderLayer)
def compute_td_grad_sample(
    layer: nn.TransformerDecoderLayer, activations: List[torch.Tensor], backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    ret = {}
    if layer.requires_grad:
        ret[layer] = layer.grad
    return ret