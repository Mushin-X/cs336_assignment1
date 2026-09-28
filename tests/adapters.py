from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    from cs336_basics.transformer.linear import Linear

    linear = Linear(d_in, d_out)
    linear.load_state_dict({'w' : weights})
    return linear(in_features)

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    from cs336_basics.transformer.embedding import Embedding

    embedding = Embedding(vocab_size, d_model)
    embedding.load_state_dict({'w' : weights})
    return embedding(token_ids)

def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    from cs336_basics.transformer.swiglu import SwiGLU

    swiglu = SwiGLU(d_model, d_ff)
    swiglu.load_state_dict({'w1':w1_weight, 'w2':w2_weight, 'w3':w3_weight})
    return swiglu(in_features)

def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    from cs336_basics.transformer.scaled_dot_product_attention import ScaledDotProductAttention

    scaled_dot_product_attention = ScaledDotProductAttention()
    return scaled_dot_product_attention(Q, K, V, mask)

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    from cs336_basics.transformer.multihead_self_attention import MultiheadSelfAttention

    multihead_self_attention = MultiheadSelfAttention(d_model, num_heads)
    multihead_self_attention.load_state_dict({'q_weight':q_proj_weight, 'k_weight':k_proj_weight, 'v_weight':v_proj_weight, 'o_weight':o_proj_weight})
    return multihead_self_attention(in_features)

def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    from cs336_basics.transformer.multihead_self_attention_with_rope import MultiheadSelfAttentionWithRoPE
    
    multihead_self_attention_with_rope  = MultiheadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, theta)
    multihead_self_attention_with_rope.load_state_dict({'q_weight':q_proj_weight, 'k_weight':k_proj_weight, 'v_weight':v_proj_weight, 'o_weight':o_proj_weight})
    return multihead_self_attention_with_rope(in_features, token_positions)

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    from cs336_basics.transformer.rope import RoPE
    rope = RoPE(theta, d_k, max_seq_len)
    return rope(in_query_or_key, token_positions)

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    from cs336_basics.transformer.transformer_block import TransformerBlock
    transformer_block = TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta)

    state_dict = {
        "multihead_att_with_rope.q_weight": weights["attn.q_proj.weight"],
        "multihead_att_with_rope.k_weight": weights["attn.k_proj.weight"],
        "multihead_att_with_rope.v_weight": weights["attn.v_proj.weight"],
        "multihead_att_with_rope.o_weight": weights["attn.output_proj.weight"],
        "rmsnorm1.g": weights["ln1.weight"],
        "rmsnorm2.g": weights["ln2.weight"],
        "swiglu.w1": weights["ffn.w1.weight"],
        "swiglu.w2": weights["ffn.w2.weight"],
        "swiglu.w3": weights["ffn.w3.weight"],
    }
    transformer_block.load_state_dict(state_dict)
    return transformer_block(in_features)

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    from cs336_basics.transformer.transformer_lm import TransformerLm

    transformer_lm = TransformerLm(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)

    #参数字典
    state_dict = {}

    # token 嵌入
    state_dict["embedding.w"] = weights["token_embeddings.weight"]

    # 每一层
    for i in range(num_layers):
        ref = f"layers.{i}"
        my = f"block.{i}"

        # 第一个 RMSNorm
        state_dict[f"{my}.rmsnorm1.g"] = weights[f"{ref}.ln1.weight"]

        # 多头注意力
        state_dict[f"{my}.multihead_att_with_rope.q_weight"] = weights[f"{ref}.attn.q_proj.weight"]
        state_dict[f"{my}.multihead_att_with_rope.k_weight"] = weights[f"{ref}.attn.k_proj.weight"]
        state_dict[f"{my}.multihead_att_with_rope.v_weight"] = weights[f"{ref}.attn.v_proj.weight"]
        state_dict[f"{my}.multihead_att_with_rope.o_weight"] = weights[f"{ref}.attn.output_proj.weight"]

        # 第二个 RMSNorm
        state_dict[f"{my}.rmsnorm2.g"] = weights[f"{ref}.ln2.weight"]
        
        # SwiGLU
        state_dict[f"{my}.swiglu.w1"] = weights[f"{ref}.ffn.w1.weight"]
        state_dict[f"{my}.swiglu.w2"] = weights[f"{ref}.ffn.w2.weight"]
        state_dict[f"{my}.swiglu.w3"] = weights[f"{ref}.ffn.w3.weight"]

    # 最终 RMSNorm
    state_dict["rmsnorm.g"] = weights["ln_final.weight"]

    # LM head
    state_dict["linear.w"] = weights["lm_head.weight"]

    transformer_lm.load_state_dict(state_dict)
    return transformer_lm(in_indices)

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    from cs336_basics.transformer.rmsnorm import RMSNorm
    rmsnorm = RMSNorm(d_model, eps)
    rmsnorm.load_state_dict({'g' : weights})
    return rmsnorm(in_features)

def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return in_features * torch.sigmoid(in_features)

def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    from cs336_basics.train_data.data_loading import DataLoading
    data_loading = DataLoading(dataset, batch_size, context_length, device)
    return data_loading.get_batch()

def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    from cs336_basics.transformer.softmax import softmax
    return softmax(in_features, dim)

def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    from cs336_basics.train.cross_entropy import cross_entropy
    return cross_entropy(inputs, targets)

def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    from cs336_basics.train.gradient_clipping import GradientClipping
    gradient_clipping = GradientClipping(parameters, max_l2_norm)
    return gradient_clipping.cal()

def get_adamw_cls() -> Any:
    from cs336_basics.train.adamw import AdamW
    return AdamW

def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    from cs336_basics.train.learning_rate_schedule import LearningRateSchedule
    learning_rate_schedule = LearningRateSchedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
    return learning_rate_schedule.get_lr()

def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    from cs336_basics.train_data.checkpointing import save_checkpoint
    save_checkpoint(model, optimizer, iteration, out)

def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    from cs336_basics.train_data.checkpointing import load_checkpoint
    return load_checkpoint(src, model, optimizer)

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    from cs336_basics.tokenizer.tokenizer import Tokenizer
    return Tokenizer(vocab, merges, special_tokens)

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    from cs336_basics.tokenizer.train_bpe import run_train_bpe as _run_train_bpe
    return _run_train_bpe(input_path, vocab_size, special_tokens, **kwargs)
