"""KV cache metadata for pruned sequences."""

from typing import Dict, Optional

import torch


class SIFTVLACache(tuple):
    """HF 4.40 cache tuples with original positions and per-layer padding masks."""

    def __new__(
        cls,
        values,
        *,
        logical_length: int,
        next_position_ids: torch.Tensor,
        layer_attention_masks: Dict[int, Optional[torch.Tensor]],
    ):
        cache = super().__new__(cls, values)
        cache.logical_length = int(logical_length)
        cache.next_position_ids = next_position_ids
        cache.layer_attention_masks = dict(layer_attention_masks)
        return cache

    def __getnewargs_ex__(self):
        return (tuple(self),), {
            "logical_length": self.logical_length,
            "next_position_ids": self.next_position_ids,
            "layer_attention_masks": self.layer_attention_masks,
        }

    def with_reordered_values(self, values, beam_idx: torch.LongTensor):
        """Reorder positions and masks with the beam's KV tensors."""
        positions = self.next_position_ids.index_select(0, beam_idx.to(self.next_position_ids.device))
        masks = {
            layer: None if mask is None else mask.index_select(0, beam_idx.to(mask.device))
            for layer, mask in self.layer_attention_masks.items()
        }
        return type(self)(
            values,
            logical_length=self.logical_length,
            next_position_ids=positions,
            layer_attention_masks=masks,
        )


def get_cache_seq_length(past_key_values, logical: bool = False) -> int:
    """Return the first layer's KV length, or the original unpruned length."""
    if past_key_values is None:
        return 0
    if logical and isinstance(past_key_values, SIFTVLACache):
        return past_key_values.logical_length
    if isinstance(past_key_values, (tuple, list)):
        return int(past_key_values[0][0].shape[-2]) if past_key_values else 0
    return int(past_key_values.get_seq_length())
