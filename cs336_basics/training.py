import numpy as np
import torch

def cross_entropy(inputs, targets):
    max_logit = torch.max(inputs, dim=1, keepdim=True)[0]
    logits_stabilized = inputs - max_logit
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_stabilized), dim=1))
    true_class_logits = logits_stabilized.gather(dim=1, index=targets.unsqueeze(1)).squeeze()
    loss_per_example = -true_class_logits + log_sum_exp
    return torch.mean(loss_per_example)