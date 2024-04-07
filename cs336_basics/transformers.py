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
    d_k = scores.size(-1)
    if mask is not None:
        scores.masked_fill_(mask, -1e9)
    softmax_scores = softmax(scores, dim=-1)
    if pdrop is not None:
        dropout = torch.nn.Dropout(p=pdrop, inplace=True)
        dropout(softmax_scores)
    attention = softmax_scores @ v
    return attention

def multi_head_self_attention(d_model: int, num_heads: int, attn_pdrop: float | None, weights: dict[str, torch.FloatTensor], in_features: torch.FloatTensor):
    batch_size, seq_length, _ = in_features.size() #_ is d_model
    d_key =  d_model // num_heads #model size divided by number of heads
    
    q_weights = torch.cat([weights[f'q_heads.{i}.weight'] for i in range(num_heads)], dim=0)
    k_weights = torch.cat([weights[f'k_heads.{i}.weight'] for i in range(num_heads)], dim=0)
    v_weights = torch.cat([weights[f'v_heads.{i}.weight'] for i in range(num_heads)], dim=0)
    output_proj_weight = weights['output_proj.weight']

    def project(x, weights):
        # x --> (batch_size, seq_length, d_model)
        # weights --> (d_key, d_model)
        projected = torch.matmul(x, weights.transpose(0, 1))
        return projected.view(batch_size, seq_length, num_heads, d_key).transpose(1, 2)

    qs = project(in_features, q_weights) 
    ks = project(in_features, k_weights)
    vs = project(in_features, v_weights) 
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1) > 0
    attn_output = scaled_dot_product_attention(qs, ks, vs, mask, pdrop=attn_pdrop)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, d_model)
    output = torch.matmul(attn_output, output_proj_weight.transpose(0, 1))
    return output