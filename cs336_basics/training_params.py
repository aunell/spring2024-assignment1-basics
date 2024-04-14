import numpy as np
import torch
from collections.abc import Callable, Iterable
from typing import Optional
import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
import time

def cross_entropy(inputs, targets):
    max_logit = torch.max(inputs, dim=1, keepdim=True)[0]
    logits_stabilized = inputs - max_logit
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_stabilized), dim=1))
    true_class_logits = logits_stabilized.gather(dim=1, index=targets.unsqueeze(1)).squeeze()
    loss_per_example = -true_class_logits + log_sum_exp
    return torch.mean(loss_per_example)

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p] # Get state associated with p.
            t = state.get("t", 0) # Get iteration number from the state, or initial value.
            grad = p.grad.data # Get the gradient of loss with respect to p.
            p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
            state["t"] = t + 1
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        self.lamda = weight_decay
        self.betas = betas
        self.eps = eps
        self.params=params


    def step(self, closure: Optional[Callable] = None):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                state = self.state[p]
                if p.grad is None:
                    continue
                if len(state)==0:
                    state['m']=torch.zeros_like(p.data)
                    state['v']=torch.zeros_like(p.data)
                m, v = state['m'], state['v']
                beta1, beta2 = self.betas
                t = state.get("t", 1)
                grad = p.grad.data 
                m = beta1*m + (1-beta1)*grad
                v = beta2*v + (1-beta2)*(grad**2)
                a_t = lr * math.sqrt(1 - beta2**(t)) / (1 - beta1**(t))
                p.data -= a_t*m / (torch.sqrt(v) +self.eps)
                p.data-= self.lamda*p.data*lr
                state['m'], state['v'] = m, v
                state['t'] = t+1

def toy_training():
    lr_list = [1,10,100]
    results={}
    for lr in lr_list:
        start = time.time()
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)
        for t in range(10):
            opt.zero_grad() # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean() # Compute a scalar loss value.
            print(loss.cpu().item())
            loss.backward() # Run backward pass, which computes gradients.
            opt.step() # Run optimizer step.
        end = time.time()
        results[str(lr)]=(end-start, loss.cpu().item())
    return results


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it<warmup_iters:
        return it/warmup_iters*max_learning_rate
    elif warmup_iters<=it<=cosine_cycle_iters:
        return min_learning_rate+.5*(1+math.cos((it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi))*(max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate
    
def gradient_clipping(parameters, max_l2_norm):
    for p in parameters:
        if p.grad is not None:
            norm = torch.norm(p.grad, p=2)
            if norm > max_l2_norm:
                p.grad*max_l2_norm/(norm+1e-6)