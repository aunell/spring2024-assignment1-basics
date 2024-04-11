vocab_size = 50257
context_length = 1024
num_layers = 48
d_model = 1600
num_heads = 25
d_ff = 6400

def calculate_params():
    # GPT-2 XL Configuration

    # Calculations
    # 1. Embeddings
    token_embeddings_params = vocab_size * d_model
    position_embeddings_params = context_length * d_model

    # 2. Transformer Layers
    # Multi-Head Self-Attention (Q, K, V matrices and output projection)
    attn_params = 3 * (d_model * d_model) + (d_model * d_model)
    # Feed-Forward Network
    ffn_params = (d_model * d_ff) + (d_ff * d_model)
    # Layer Normalization (scale and bias for 2 LayerNorm layers)
    # ln_params = 2 * 2 * d_model
    # Total parameters per Transformer layer
    transformer_layer_params = attn_params + ffn_params #+ ln_params
    # Total for all Transformer layers
    transformer_layers_params = num_layers * transformer_layer_params

    # 3. Total Parameters
    total_params = token_embeddings_params + position_embeddings_params + transformer_layers_params

    # Memory Required (4 bytes per parameter, for single-precision floating-point)
    memory_bytes = total_params * 4
    memory_gigabytes = memory_bytes / (1024 ** 3)

    return (total_params, memory_gigabytes)

def calculate_flops():
    # FLOPs for Q, K, V projections in a single transformer layer
    flops_qkv = 3 * (context_length * d_model * d_model)

    # FLOPs for dot-product attention (assuming context_length tokens, batch size of 1, and simplified calculation)
    # Here, the output size is the same as the input, and we calculate the attention mechanism's complexity.
    flops_attention = (context_length * (d_model) * context_length) 

    # FLOPs for output projection of the attention mechanism
    flops_attn_output_proj = context_length * d_model * d_model

    # FLOPs for the feed-forward network within a single transformer layer
    flops_ffn = 2 * (context_length * d_model * d_ff)

    # Total FLOPs per transformer layer
    flops_per_layer = flops_qkv + flops_attention + flops_attn_output_proj + flops_ffn

    # Total FLOPs for all transformer layers
    total_flops = flops_per_layer * num_layers

    # Convert to gigaFLOPs for readability
    total_gflops = total_flops / 1e9

    return(total_gflops)

print(calculate_flops())
