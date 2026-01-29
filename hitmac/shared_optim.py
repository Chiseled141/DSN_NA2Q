"""
=============================================================================
SHARED OPTIMIZERS FOR A3C (Multi-Process Training)
=============================================================================

WHAT IS THIS FILE?
------------------
This file implements shared-memory optimizers that enable distributed 
training across multiple CPU/GPU processes. These are essential components
of the A3C (Asynchronous Advantage Actor-Critic) algorithm.

WHY SHARED MEMORY?
------------------
In A3C, multiple worker processes train simultaneously:

    Worker 1 ──┐
    Worker 2 ──┼──► Shared Model ◄──► Shared Optimizer
    Worker 3 ──┘
    
Each worker computes gradients locally, but the optimizer statistics
(momentum buffers, etc.) must be shared across all workers for 
consistent updates.

ARCHITECTURE:
-------------
    ┌─────────────────────────────────────────────────────────┐
    │                    SHARED MEMORY                         │
    │  ┌─────────────────┐  ┌─────────────────┐               │
    │  │   exp_avg       │  │   exp_avg_sq    │  (Adam)       │
    │  │  (momentum 1)   │  │  (momentum 2)   │               │
    │  └─────────────────┘  └─────────────────┘               │
    │  ┌─────────────────┐                                    │
    │  │   square_avg    │  (RMSprop)                         │
    │  └─────────────────┘                                    │
    └─────────────────────────────────────────────────────────┘
                    ▲           ▲           ▲
                    │           │           │
              ┌─────┴───┐ ┌─────┴───┐ ┌─────┴───┐
              │Worker 1 │ │Worker 2 │ │Worker 3 │
              └─────────┘ └─────────┘ └─────────┘

OPTIMIZERS:
-----------
- SharedAdam:   Adam optimizer with AMSGrad support
- SharedRMSprop: RMSprop optimizer with momentum support
"""

import torch
import torch.optim as optim


class SharedAdam(optim.Adam):
    """
    Adam optimizer with shared state for multi-process A3C training.
    
    Shares momentum buffers (exp_avg, exp_avg_sq) across processes
    using PyTorch's shared memory tensors, enabling asynchronous updates.
    
    Parameters:
    -----------
    params : iterable
        Model parameters to optimize
    lr : float
        Learning rate (default: 1e-3)
    betas : Tuple[float, float]
        Coefficients for running averages (default: (0.9, 0.999))
    eps : float
        Term added for numerical stability (default: 1e-8)
    weight_decay : float
        Weight decay / L2 regularization (default: 0)
    amsgrad : bool
        Whether to use AMSGrad variant (default: False)
    
    Example:
    --------
    >>> model = MyModel().share_memory()
    >>> optimizer = SharedAdam(model.parameters(), lr=0.0005, amsgrad=True)
    >>> optimizer.share_memory()
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        super(SharedAdam, self).__init__(
            params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad
        )
        # Initialize shared state for all parameters
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1).share_memory_()
                state['exp_avg'] = torch.zeros_like(p.data).share_memory_()
                state['exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()
                if amsgrad:
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()

    def share_memory(self):
        """Explicitly share memory (called after initialization)."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                if 'max_exp_avg_sq' in state:
                    state['max_exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Parameters:
        -----------
        closure : callable, optional
            A closure that reevaluates the model and returns the loss
        
        Returns:
        --------
        loss : float or None
            Loss value if closure provided
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                amsgrad = group.get('amsgrad', False)
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay running averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class SharedRMSprop(optim.RMSprop):
    """
    RMSprop optimizer with shared state for multi-process A3C training.
    
    Shares gradient statistics (square_avg) across processes using
    PyTorch's shared memory tensors.
    
    Parameters:
    -----------
    params : iterable
        Model parameters to optimize
    lr : float
        Learning rate (default: 1e-2)
    alpha : float
        Smoothing constant (default: 0.99)
    eps : float
        Term added for numerical stability (default: 1e-8)
    weight_decay : float
        Weight decay / L2 regularization (default: 0)
    momentum : float
        Momentum factor (default: 0)
    centered : bool
        If True, compute centered RMSprop (default: False)
    """
    
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8,
                 weight_decay=0, momentum=0, centered=False):
        super(SharedRMSprop, self).__init__(
            params, lr=lr, alpha=alpha, eps=eps,
            weight_decay=weight_decay, momentum=momentum, centered=centered
        )
        # Initialize shared state for all parameters
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1).share_memory_()
                state['square_avg'] = torch.zeros_like(p.data).share_memory_()
                if momentum > 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data).share_memory_()
                if centered:
                    state['grad_avg'] = torch.zeros_like(p.data).share_memory_()

    def share_memory(self):
        """Explicitly share memory (called after initialization)."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['square_avg'].share_memory_()
                if 'momentum_buffer' in state:
                    state['momentum_buffer'].share_memory_()
                if 'grad_avg' in state:
                    state['grad_avg'].share_memory_()

    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Parameters:
        -----------
        closure : callable, optional
            A closure that reevaluates the model and returns the loss
        
        Returns:
        --------
        loss : float or None
            Loss value if closure provided
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                state = self.state[p]
                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                # Update running average of squared gradient
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.data.add_(buf, alpha=-group['lr'])
                else:
                    p.data.addcdiv_(grad, avg, value=-group['lr'])

        return loss
