"""SIFT-VLA pruning during Llama prefill."""

from typing import Optional, Tuple

import torch

from rd_token.methods.sift_vla import sift_vla_prune


def apply_sift_vla_if_needed(
    llama_model,
    hidden_states: torch.Tensor,
    layer_idx: int,
    position_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: Optional[torch.Tensor],
    past_seen_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], bool]:
    """Prune at the configured layer and retain original RoPE positions."""

    cfg = llama_model.config

    if not getattr(cfg, "sift_vla_enable", False):
        return hidden_states, position_ids, None, cache_position, False

    sift_vla_layer = int(getattr(cfg, "sift_vla_layer", 3))
    sift_vla_vision_start = int(getattr(cfg, "sift_vla_vision_start", 1))
    sift_vla_num_visual_tokens = int(getattr(cfg, "sift_vla_num_visual_tokens", 256))

    if int(layer_idx) != sift_vla_layer:
        return hidden_states, position_ids, None, cache_position, False

    past_seen = int(past_seen_tokens)
    seq_len = int(hidden_states.shape[1])

    if past_seen != 0 or seq_len == 1:
        return hidden_states, position_ids, None, cache_position, False

    if seq_len <= sift_vla_vision_start + sift_vla_num_visual_tokens:
        return hidden_states, position_ids, None, cache_position, False

    if cache_position is not None and (
        cache_position.numel() != seq_len or int(cache_position[0]) != 0
    ):
        return hidden_states, position_ids, None, cache_position, False

    result = sift_vla_prune(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        cfg=cfg,
    )

    if not result.applied:
        return hidden_states, position_ids, None, cache_position, False

    hidden_states = result.hidden_states
    new_seq_length = int(hidden_states.shape[1])

    keep_indices = result.keep_indices
    if keep_indices is None:
        raise ValueError("SIFT-VLA selection must return keep_indices for the original sequence.")
    keep_indices = keep_indices.to(device=hidden_states.device, dtype=torch.long)
    if attention_mask is not None:
        attention_mask = attention_mask.index_select(-1, keep_indices.to(attention_mask.device))

    # RoPE uses original positions, not compact KV indices.
    position_ids = position_ids.index_select(-1, keep_indices.to(position_ids.device))

    cache_position = torch.arange(
        0,
        new_seq_length,
        device=hidden_states.device,
        dtype=torch.long,
    )

    return hidden_states, position_ids, attention_mask, cache_position, True
