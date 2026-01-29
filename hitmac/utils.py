"""
=============================================================================
HiT-MAC UTILITY FUNCTIONS
=============================================================================

WHAT IS THIS FILE?
------------------
Contains utility functions for HiT-MAC model initialization and processing.

FUNCTIONS:
----------
- norm_col_init():    Normalize weight matrix columns
- weights_init():     Xavier/He initialization for Conv and Linear layers
- goal_id_filter():   Filter goal IDs from tracking matrix
"""

import numpy as np
import torch
import torch.nn as nn


def norm_col_init(weights, std=1.0):
    """
    Normalize columns of weight matrix.
    
    Parameters:
    -----------
    weights : torch.Tensor
        Weight matrix to normalize
    std : float
        Target standard deviation (default: 1.0)
    
    Returns:
    --------
    torch.Tensor
        Normalized weight matrix
    """
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x ** 2).sum(1, keepdim=True))
    return x


def weights_init(m):
    """
    Initialize network weights using Xavier/He initialization.
    
    Applies proper initialization to Conv and Linear layers based on
    fan-in and fan-out to maintain gradient flow during training.
    
    Parameters:
    -----------
    m : nn.Module
        Module to initialize (usually called via model.apply())
    
    Example:
    --------
    >>> model = MyModel()
    >>> model.apply(weights_init)
    """
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


def goal_id_filter(goals):
    """
    Filter goal IDs where tracking confidence > 0.5.
    
    Used to identify which targets are being actively tracked
    by a sensor based on the goal assignment matrix.
    
    Parameters:
    -----------
    goals : np.ndarray
        Goal assignment vector (values 0-1)
    
    Returns:
    --------
    np.ndarray
        Indices where goals > 0.5
    """
    return np.where(goals > 0.5)[0]
