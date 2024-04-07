import torch.nn as nn
import torch
import math
import numpy as np

def rms_norm(in_features, weights, eps=1e-8):
    weight = nn.Parameter(weights['weight'])
    rms = torch.sqrt(torch.mean(in_features ** 2, dim=-1, keepdim=True) + eps)
    normalized_features = in_features / rms
    final_output = weight * normalized_features
    return final_output

def gelu(in_features: torch.FloatTensor):
    ans = .5 * in_features
    ans1 = (1 + torch.erf(in_features/(torch.sqrt(torch.tensor(2.0)))))
    return ans*ans1

def feedforward(d_model: int, d_ff: int, in_features: torch.FloatTensor, weights: dict[str, torch.FloatTensor]):
    return gelu(in_features@weights['w1.weight'].t())@weights['w2.weight'].t()

def softmax(in_features: torch.FloatTensor, dim: int):
    in_features = in_features - torch.max(in_features, dim=dim, keepdim=True)[0]
    return torch.exp(in_features) / torch.sum(torch.exp(in_features), dim=dim, keepdim=True)

def scaled_dot_product_attention(q: torch.FloatTensor, k: torch.FloatTensor, v: torch.FloatTensor, mask: None, pdrop: None):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask==None:
        mask = torch.ones((d_k, d_k), dtype=torch.bool)
    scores.masked_fill_(mask, -1e9)
    softmax_scores = softmax(scores, dim=-1)
    attention = softmax_scores @ v
    return attention