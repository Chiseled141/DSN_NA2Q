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
"""

import numpy as np
import torch
import torch.nn as nn


def scripted_action(env, sensor_idx, goal):
    """
    Scripted executor policy (paper Appendix B).

    Points sensor toward the center of its assigned targets.
    Used during coordinator training for speed (75 FPS vs 25 FPS).

    Args:
        env: DSNEnv instance (access to internal state)
        sensor_idx: which sensor
        goal: [n_targets] binary array — which targets are assigned

    Returns:
        int: 0=TurnLeft, 1=Stay, 2=TurnRight
    """
    assigned = np.where(goal > 0)[0]
    if len(assigned) == 0:
        return 1  # Stay

    # Center of assigned targets (paper Appendix B)
    center = env.target_positions[assigned].mean(axis=0)
    sensor_pos = env.sensor_positions[sensor_idx]
    sensor_angle = env.sensor_angles[sensor_idx]

    dx, dy = center[0] - sensor_pos[0], center[1] - sensor_pos[1]
    target_angle = np.arctan2(dy, dx)
    angle_error = ((target_angle - sensor_angle + np.pi) % (2 * np.pi)) - np.pi

    # action = clip(error // z_delta, -1, 1), z_delta = 5°
    z_delta = np.radians(5.0)
    val = angle_error / z_delta
    if val > 0.5:
        return 2   # TurnRight
    elif val < -0.5:
        return 0   # TurnLeft
    return 1       # Stay


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
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
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

    if classname.find("Conv") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1] * weight_shape[2] * weight_shape[3]
        fan_out = weight_shape[0] * weight_shape[2] * weight_shape[3]
        w_bound = torch.sqrt(torch.tensor(6.0 / (fan_in + fan_out)))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif classname.find("Linear") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = torch.sqrt(torch.tensor(6.0 / (fan_in + fan_out)))
        m.weight.data.uniform_(-w_bound, w_bound)
        if m.bias is not None:
            m.bias.data.fill_(0)


