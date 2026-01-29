"""
HiT-MAC Model Utilities - Initialization and weight normalization.
"""

import torch
import torch.nn as nn


def norm_col_init(weights, std=1.0):
    """Normalize columns of weight matrix."""
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x ** 2).sum(1, keepdim=True))
    return x


def weights_init(m):
    """Initialize network weights using orthogonal initialization."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1] * weight_shape[2] * weight_shape[3]
        fan_out = weight_shape[0] * weight_shape[2] * weight_shape[3]
        w_bound = torch.sqrt(torch.tensor(6. / (fan_in + fan_out)))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = torch.sqrt(torch.tensor(6. / (fan_in + fan_out)))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)
