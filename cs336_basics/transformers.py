from collections.abc import Callable, Iterable
from typing import Optional
import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
import time

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

def multi_head_self_attention(d_model: int, num_heads: int, attn_pdrop: float | None, weights: dict[str, torch.FloatTensor], in_features: torch.FloatTensor, weight_keys: dict[str, str]):
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
        q_weights = weights[weight_keys['q_proj']]
        k_weights = weights[weight_keys['k_proj']]
        v_weights = weights[weight_keys['v_proj']]
        output_proj_weight = weights[weight_keys['attn.output_proj']]

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

def transformer_block(d_model: int,
    num_heads: int,
    d_ff: int,
    attn_pdrop: float,
    residual_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
    weight_keys: dict):
    dropout = torch.nn.Dropout(p=residual_pdrop, inplace=True)
    rms_norm_output = rms_norm(in_features, weights, key=weight_keys['rms_norm_1'])
    multi_head_self_attention_output = multi_head_self_attention(d_model, num_heads, attn_pdrop, weights, rms_norm_output, weight_keys=weight_keys)
    dropout(multi_head_self_attention_output)
    attn_out = in_features + multi_head_self_attention_output

    rms_norm_output = rms_norm(attn_out, weights, key=weight_keys['rms_norm_2'])
    feedforward_output = feedforward(d_model, d_ff, rms_norm_output, weights, weight_keys['positionwise_feedforward_1'], weight_keys['positionwise_feedforward_2'])
    dropout(feedforward_output)

    output = attn_out + feedforward_output
    return output

def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    attn_pdrop: float,
    residual_pdrop: float,
    weights: dict[str, torch.FloatTensor],
    in_indices: torch.LongTensor,
):
    token_embeddings= weights['token_embeddings.weight'][in_indices] # (vocab_size, d_model).
    position_ids = torch.arange(in_indices.shape[1]).repeat(in_indices.shape[0], 1)
    position_embeddings = weights['position_embeddings.weight'][position_ids]
    in_features = token_embeddings + position_embeddings
    input_embeddings = F.dropout(in_features, p=residual_pdrop, inplace=False)
    hidden_states = input_embeddings
    for layer_number in range(num_layers):
        weight_keys = {
            "rms_norm_1": f"layers.{layer_number}.ln1.weight",
            "rms_norm_2": f"layers.{layer_number}.ln2.weight",
            "positionwise_feedforward_1": f"layers.{layer_number}.ffn.w1.weight",
            "positionwise_feedforward_2": f"layers.{layer_number}.ffn.w2.weight",
            "q_proj": f"layers.{layer_number}.attn.q_proj.weight",
            "k_proj": f"layers.{layer_number}.attn.k_proj.weight",
            "v_proj": f"layers.{layer_number}.attn.v_proj.weight",
            "attn.output_proj": f"layers.{layer_number}.attn.output_proj.weight"
        }
        hidden_states = transformer_block(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop, weights, hidden_states, weight_keys=weight_keys)
    normalized_input_embeddings = rms_norm(hidden_states, weights, key='ln_final.weight')
    linear = torch.nn.functional.linear(normalized_input_embeddings, weights['lm_head.weight'])
    # output = softmax(linear, dim=-1)
    return linear

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

#model class subclass of nn.module with transformer block 
                ##transformer lm model class has a list of transformer block classes (which also stores weights )