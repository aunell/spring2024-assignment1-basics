from collections.abc import Callable, Iterable
from typing import Optional
import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
import time

class Transformer_LM(nn.Module):
    def __init__(self, weights, d_model, num_layers, num_heads, d_ff, attn_pdrop, residual_pdrop, vocab_size, context_length, in_indices):
        super(Transformer_LM, self).__init__()
        self.weights = weights
        self.weights_for_layer = self.normalize_weights(weights, num_layers)
        self.in_indices = in_indices
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
    
    def forward(self):
        token_embeddings= self.weights['token_embeddings.weight'][self.in_indices] # (vocab_size, d_model).
        position_ids = torch.arange(self.in_indices.shape[1]).repeat(self.in_indices.shape[0], 1)
        position_embeddings = self.weights['position_embeddings.weight'][position_ids]
        in_features = token_embeddings + position_embeddings
        input_embeddings = F.dropout(in_features, p=self.residual_pdrop, inplace=False)
        hidden_states = input_embeddings
        for layer_number in range(self.num_layers):
            # breakpoint()
            transformer = Transformer_block(self.d_model, self.num_heads, self.d_ff, self.attn_pdrop, self.residual_pdrop, self.weights_for_layer[layer_number], hidden_states)
            hidden_states = transformer.forward()
        # breakpoint()
        normalized_input_embeddings = rms_norm(hidden_states, self.weights, key='ln_final.weight')
        linear = torch.nn.functional.linear(normalized_input_embeddings, self.weights['lm_head.weight'])
        return linear
    
    def normalize_weights(self, weights, num_layers):
        layer_weights={}
        for layer_number in range(num_layers):
            layer_weights[layer_number] = {
                "ln1.weight": weights[f"layers.{layer_number}.ln1.weight"],
                "ln2.weight": weights[f"layers.{layer_number}.ln2.weight"],
                "ffn.w1.weight": weights[f"layers.{layer_number}.ffn.w1.weight"],
                "ffn.w2.weight": weights[f"layers.{layer_number}.ffn.w2.weight"],
                "attn.q_proj.weight": weights[f"layers.{layer_number}.attn.q_proj.weight"],
                "attn.k_proj.weight": weights[f"layers.{layer_number}.attn.k_proj.weight"],
                "attn.v_proj.weight": weights[f"layers.{layer_number}.attn.v_proj.weight"],
                "attn.output_proj.weight": weights[f"layers.{layer_number}.attn.output_proj.weight"]
            }
        return layer_weights
    
def rms_norm(in_features, weights, eps=1e-5, key='weight'):
    weight = nn.Parameter(weights[key])
    rms = torch.sqrt(torch.mean(in_features ** 2, dim=-1, keepdim=True) + eps)
    normalized_features = in_features / rms
    final_output = weight * normalized_features
    return final_output

def gelu(in_features: torch.FloatTensor):
    ans = .5 * in_features
    ans1 = (1 + torch.erf(in_features/(torch.sqrt(torch.tensor(2.0)))))
    return ans*ans1

def feedforward(d_model: int, d_ff: int, in_features: torch.FloatTensor, weights: dict[str, torch.FloatTensor], w1: str, w2: str):
    return gelu(in_features@weights[w1].t())@weights[w2].t()

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
    
    try:
        #for testing
        q_weights = torch.cat([weights[f'q_heads.{i}.weight'] for i in range(num_heads)], dim=0)
        k_weights = torch.cat([weights[f'k_heads.{i}.weight'] for i in range(num_heads)], dim=0)
        v_weights = torch.cat([weights[f'v_heads.{i}.weight'] for i in range(num_heads)], dim=0)
        output_proj_weight = weights['output_proj.weight']
    except:
        #for transformer block
        q_weights = weights['attn.q_proj.weight']
        k_weights = weights['attn.k_proj.weight']
        v_weights = weights['attn.v_proj.weight']
        output_proj_weight = weights['attn.output_proj.weight']

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

class Transformer_block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, attn_pdrop, residual_pdrop, weights, in_features):
        super(Transformer_block, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
        self.weights = weights
        self.in_features = in_features
    
    def forward(self):
        dropout = torch.nn.Dropout(p=self.residual_pdrop, inplace=True)
        rms_norm_output = rms_norm(self.in_features, self.weights, key='ln1.weight')
        multi_head_self_attention_output = multi_head_self_attention(self.d_model, self.num_heads, self.attn_pdrop, self.weights, rms_norm_output)
        dropout(multi_head_self_attention_output)
        attn_out = self.in_features + multi_head_self_attention_output

        rms_norm_output = rms_norm(attn_out, self.weights, key='ln2.weight')
        feedforward_output = feedforward(self.d_model, self.d_ff, rms_norm_output, self.weights, 'ffn.w1.weight', 'ffn.w2.weight')
        dropout(feedforward_output)

        output = attn_out + feedforward_output
        return output