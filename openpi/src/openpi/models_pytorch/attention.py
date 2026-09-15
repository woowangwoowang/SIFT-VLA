"""Exact attention for pi0's bidirectional prefix and block-causal action stream.

FlashAttention's ordinary causal flag does not express this policy's mask.  We
pack queries with identical visible keys into independent noncausal sequences;
FlashAttention then computes exactly the same softmax over each allowed set.
"""

from collections import OrderedDict
import dataclasses
from functools import lru_cache
import os

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

# Strong references prevent tensor-id reuse. The small bound prevents retaining
# masks from an unbounded number of policy requests.
_MASK_PLANS: OrderedDict = OrderedDict()
_MAX_MASK_PLANS = 8


@dataclasses.dataclass
class _MaskPlan:
    query_indices: torch.Tensor
    key_indices: torch.Tensor
    cu_queries: torch.Tensor
    cu_keys: torch.Tensor
    max_queries: int
    max_keys: int
    uniform_query_indices: torch.Tensor
    uniform_batch_indices: torch.Tensor


@lru_cache(maxsize=1)
def _flash_function():
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: PLC0415 (optional CUDA extension)
    except (ImportError, OSError):
        return None
    return flash_attn_varlen_func


def _repeat_kv(states: torch.Tensor, heads: int) -> torch.Tensor:
    groups = heads // states.shape[1]
    if groups == 1:
        return states
    batch, kv_heads, length, width = states.shape
    return states[:, :, None].expand(batch, kv_heads, groups, length, width).reshape(batch, heads, length, width)


def _mask_plan(mask: torch.Tensor | None, query: torch.Tensor, key: torch.Tensor) -> _MaskPlan | None:
    batch, _, queries, _ = query.shape
    keys = key.shape[-2]
    # Inference tensors have no mutation counter, so do not cache their plans.
    # Normal no_grad policy execution keeps counters and reuses each plan.
    cacheable = True
    try:
        version = mask._version if mask is not None else None  # noqa: SLF001
    except RuntimeError:
        version = None
        cacheable = False
    cache_key = (id(mask), version, batch, queries, keys, query.device, query.dtype)
    cached = _MASK_PLANS.get(cache_key) if cacheable else None
    if cached is not None and cached[0] is mask:
        _MASK_PLANS.move_to_end(cache_key)
        return cached[1]

    if mask is None:
        allowed = np.ones((batch, queries, keys), dtype=bool)
        finite_blocked = False
    else:
        # A separate mask per head or a learnable bias needs the general SDPA
        # path. Only hard visibility masks can be expressed by token packing.
        if mask.requires_grad or mask.ndim != 4 or mask.shape[1] != 1 or mask.shape[-1] < keys:
            return None
        if mask.shape[0] not in (1, batch) or mask.shape[-2] not in (1, queries):
            return None
        effective_dtype = query.dtype if mask.is_floating_point() else mask.dtype
        values = mask[:, 0, :, :keys].detach().to(dtype=effective_dtype).to(device="cpu", dtype=torch.float32).numpy()
        values = np.broadcast_to(values, (batch, queries, keys))
        if mask.dtype == torch.bool:
            allowed = values.astype(bool)
            finite_blocked = False
        else:
            blocked = values != 0
            blocked_values = values[blocked]
            # Preserve finite additive biases by falling back, rather than
            # silently treating every negative value as an invisible token.
            if np.any(~(np.isneginf(blocked_values) | (blocked_values <= -1e30))):
                return None
            if blocked_values.size and not np.all(blocked_values == blocked_values[0]):
                return None
            allowed = ~blocked
            finite_blocked = bool(blocked_values.size and np.isfinite(blocked_values[0]))

    query_indices, key_indices = [], []
    query_lengths, key_lengths = [], []
    uniform_queries, uniform_batches = [], []
    for batch_idx in range(batch):
        packed = np.packbits(allowed[batch_idx], axis=-1)
        _, inverse = np.unique(packed, axis=0, return_inverse=True)
        # A token-by-token causal mask has many distinct rows; packing it
        # duplicates quadratically many keys. SDPA handles such masks directly.
        num_groups = int(inverse.max()) + 1
        if num_groups > 16:
            return None
        for group in range(num_groups):
            q_ids = np.flatnonzero(inverse == group)
            k_ids = np.flatnonzero(allowed[batch_idx, q_ids[0]])
            if k_ids.size == 0:
                # The legacy finite sentinel gives a uniform softmax on a
                # fully padded query. Preserve it; boolean/-inf masks give 0.
                if finite_blocked:
                    uniform_queries.extend((q_ids + batch_idx * queries).tolist())
                    uniform_batches.extend([batch_idx] * len(q_ids))
                continue
            query_indices.extend((q_ids + batch_idx * queries).tolist())
            key_indices.extend((k_ids + batch_idx * keys).tolist())
            query_lengths.append(len(q_ids))
            key_lengths.append(len(k_ids))

    def tensor(values, dtype=torch.long):
        return torch.tensor(values, dtype=dtype, device=query.device)

    plan = _MaskPlan(
        query_indices=tensor(query_indices),
        key_indices=tensor(key_indices),
        cu_queries=tensor(np.cumsum([0, *query_lengths]), dtype=torch.int32),
        cu_keys=tensor(np.cumsum([0, *key_lengths]), dtype=torch.int32),
        max_queries=max(query_lengths, default=0),
        max_keys=max(key_lengths, default=0),
        uniform_query_indices=tensor(uniform_queries),
        uniform_batch_indices=tensor(uniform_batches),
    )
    if cacheable:
        _MASK_PLANS[cache_key] = (mask, plan)
        _MASK_PLANS.move_to_end(cache_key)
        while len(_MASK_PLANS) > _MAX_MASK_PLANS:
            _MASK_PLANS.popitem(last=False)
    return plan


def _flash_attention(query, key, value, mask, *, scaling, dropout, flash_function, plan=None):
    if plan is None:
        plan = _mask_plan(mask, query, key)
    if plan is None:
        return None
    batch, heads, queries, width = query.shape
    keys = key.shape[-2]
    q_flat = query.transpose(1, 2).reshape(batch * queries, heads, width)
    k_flat = key.transpose(1, 2).reshape(batch * keys, key.shape[1], width)
    v_flat = value.transpose(1, 2).reshape(batch * keys, value.shape[1], width)
    output = torch.zeros_like(q_flat)
    if plan.query_indices.numel():
        packed_output = flash_function(
            q_flat.index_select(0, plan.query_indices),
            k_flat.index_select(0, plan.key_indices),
            v_flat.index_select(0, plan.key_indices),
            plan.cu_queries,
            plan.cu_keys,
            plan.max_queries,
            plan.max_keys,
            dropout_p=dropout,
            softmax_scale=scaling,
            causal=False,
        )
        output = output.index_copy(0, plan.query_indices, packed_output)
    if plan.uniform_query_indices.numel():
        means = _repeat_kv(value, heads).mean(dim=-2)
        output = output.index_copy(
            0, plan.uniform_query_indices, means.index_select(0, plan.uniform_batch_indices)
        )
    return output.reshape(batch, queries, heads, width)


def attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Dispatch without changing the caller's visibility mask, RoPE or scaling."""
    backend = os.environ.get("OPENPI_ATTENTION_IMPLEMENTATION", "auto").lower()
    if backend not in {"auto", "eager", "sdpa", "flash_attention_2"}:
        raise ValueError(f"Unknown OPENPI_ATTENTION_IMPLEMENTATION={backend!r}")
    want_weights = bool(kwargs.get("output_attentions", False))
    actual_backend = "eager" if want_weights or backend == "eager" else "sdpa"
    output = None
    eligible = query.is_cuda and query.dtype in (torch.float16, torch.bfloat16) and query.shape[-1] <= 256
    # Retain exact dropout semantics for fully masked finite-sentinel rows by
    # using the general implementation during training with attention dropout.
    if backend in {"auto", "flash_attention_2"} and eligible and not want_weights and dropout == 0:
        flash_function = _flash_function()
        if flash_function is not None:
            plan = _mask_plan(attention_mask, query, key)
            if plan is not None:
                output = _flash_attention(
                    query, key, value, attention_mask, scaling=scaling, dropout=dropout,
                    flash_function=flash_function, plan=plan,
                )
    if backend == "flash_attention_2" and output is None:
        raise RuntimeError(
            "FlashAttention-2 requires its installed CUDA extension, CUDA fp16/bf16 tensors with head_dim <= 256, "
            "and a hard visibility mask (without attention-weight output or attention dropout). "
            "Use OPENPI_ATTENTION_IMPLEMENTATION=auto or sdpa for the exact general-mask fallback."
        )

    weights = None
    if output is None:
        key = _repeat_kv(key, query.shape[1])
        value = _repeat_kv(value, query.shape[1])
        mask = attention_mask
        if mask is not None:
            mask = mask[..., : key.shape[-2]]
            if mask.is_floating_point():
                mask = mask.to(dtype=query.dtype)
        if actual_backend == "eager":
            logits = torch.matmul(query, key.transpose(-2, -1)) * scaling
            if mask is not None:
                logits = logits.masked_fill(~mask, -torch.inf) if mask.dtype == torch.bool else logits + mask
            weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
            weights = torch.nan_to_num(weights, nan=0.0)
            weights = F.dropout(weights, p=dropout, training=module.training)
            output = torch.matmul(weights, value).transpose(1, 2).contiguous()
        else:
            output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=dropout, is_causal=False, scale=scaling
            ).transpose(1, 2).contiguous()
    return output, weights
