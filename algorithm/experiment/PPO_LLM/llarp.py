import json
import numpy as np
import torch
import torch.nn.functional as F


def load_safetensors(filename, device, new_dtype=None):
    dtypes = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
    }

    state_dict = {}

    with open(filename, "r+b") as f:
        header_size = int.from_bytes(f.read(8), byteorder="little")
        header = f.read(header_size).decode("utf-8")
        info = json.loads(header)
        after_header = f.tell()
        m = np.memmap(f)
        for name, value in info.items():
            if name.startswith("__"): continue
            dtype = dtypes[value["dtype"]]
            shape = value["shape"]
            start, end = value["data_offsets"]
            weights = m[after_header + start:after_header + end]
            weights = torch.from_numpy(weights).view(dtype).reshape(shape)
            if new_dtype is not None:
                weights = weights.type(new_dtype)
            state_dict[name] = weights.to(device)

    return state_dict


def precompute(device):
    # Compute rotary embedding
    max_seq_len = 2048
    base = 10000
    dim = 64
    t = torch.arange(max_seq_len, device=device)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().view(max_seq_len, dim)
    sin = emb.sin().view(max_seq_len, dim)

    # Create attention mask matrix like:
    # [0, -inf, -inf, -inf]
    # [0,    0, -inf, -inf]
    # [0,    0,    0, -inf]
    # [0,    0,    0,    0]
    attention_mask = torch.triu(torch.full((max_seq_len, max_seq_len), -float("inf"), device=device), 1)

    return cos, sin, attention_mask


def embedding(token_ids, state_dict):
    x = state_dict["model.embed_tokens.weight"][token_ids]
    return x


def llama(x, position_ids, cache, state_dict):
    for i in range(22):
        # Normalize, attention, normalize again, multi-layer perceptron
        x_norm = rmsnorm(x, state_dict[f"model.layers.{i}.input_layernorm.weight"])

        x = x + attention(x_norm, position_ids, i, cache, state_dict)

        x_norm = rmsnorm(x, state_dict[f"model.layers.{i}.post_attention_layernorm.weight"])

        x = x + mlp(x_norm, i, state_dict)

    # Normalize and final projection head
    x = rmsnorm(x, state_dict["model.norm.weight"])

    return x[:, -1]


def mlp(x, i, state_dict):
    """Multi-layer perceptron."""
    gate_proj = state_dict[f"model.layers.{i}.mlp.gate_proj.weight"]
    up_proj = state_dict[f"model.layers.{i}.mlp.up_proj.weight"]
    down_proj = state_dict[f"model.layers.{i}.mlp.down_proj.weight"]

    x = F.silu(x @ gate_proj.T) * (x @ up_proj.T)
    x = x @ down_proj.T

    return x


def rmsnorm(x, weight, eps=1e-5):
    """Root mean square layer normalization."""
    dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = weight * x
    return x.to(dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def attention(x, position_ids, i, cache, state_dict):
    """Multi-head self-attention."""
    # For a detailed explanation of attention, see this video by Andrej Karpathy:
    # https://www.youtube.com/watch?v=kCc8FmEb1nY&t=1h2m
    bsz, q_len, _ = x.shape
    dtype = x.dtype
    num_heads = 32
    head_dim = 64
    num_key_value_heads = 4
    hidden_size = 2048
    num_key_value_groups = 8

    if "precomputed" not in cache:
        cache["precomputed"] = precompute(x.device)

    cos, sin, attention_mask = cache["precomputed"]

    q_proj = state_dict[f"model.layers.{i}.self_attn.q_proj.weight"]
    k_proj = state_dict[f"model.layers.{i}.self_attn.k_proj.weight"]
    v_proj = state_dict[f"model.layers.{i}.self_attn.v_proj.weight"]
    o_proj = state_dict[f"model.layers.{i}.self_attn.o_proj.weight"]

    query_states = (x @ q_proj.T).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = (x @ k_proj.T).view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = (x @ v_proj.T).view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

    # Apply rotary embedding
    partial_cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    partial_sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    query_states = (query_states * partial_cos) + (rotate_half(query_states) * partial_sin)
    key_states = (key_states * partial_cos) + (rotate_half(key_states) * partial_sin)

    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)

    if f"kv_states_{i}" in cache:
        cached_key_states, cached_value_states = cache[f"kv_states_{i}"]

        key_states = torch.cat([cached_key_states, key_states], dim=2)
        value_states = torch.cat([cached_value_states, value_states], dim=2)

    cache[f"kv_states_{i}"] = (key_states, value_states)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * head_dim ** -0.5

    # To predict future tokens, only previous tokens may be used.
    # This is ensured by weighting future tokens very negatively,
    # so they are not chosen by the softmax.
    attn_weights = attn_weights + attention_mask[position_ids, :attn_weights.shape[3]].unsqueeze(1)

    attn_weights = F.softmax(attn_weights, dim=3, dtype=torch.float32).to(dtype)

    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, hidden_size)

    attn_output = attn_output @ o_proj.T

    return attn_output

