"""SIFT-VLA visual token selection (Eqs. 1--8, Algorithm 1).

J = delta_R * R + delta_D * D. Zero total gap selects argmax R.
Min-max normalization adds epsilon to the max-minus-min denominator.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn.functional as F

from rd_token.types import SIFTVLAResult


try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
    _TRITON_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as exc:
    triton = None
    tl = None
    _HAS_TRITON = False
    _TRITON_IMPORT_ERROR = exc


_TRITON_DEVICE_SUPPORT: dict[int, tuple[bool, str]] = {}


def _triton_device_supported(tensor: torch.Tensor) -> tuple[bool, str]:
    """Reject architectures unsupported by the bundled Triton toolchain."""
    if not _HAS_TRITON or not tensor.is_cuda:
        return False, "Triton and CUDA are required"

    device_index = tensor.device.index
    if device_index is None:
        device_index = int(torch.cuda.current_device())
    cached = _TRITON_DEVICE_SUPPORT.get(int(device_index))
    if cached is not None:
        return cached

    capability = torch.cuda.get_device_capability(device_index)
    triton_major = int(triton.__version__.split(".", 1)[0])

    if int(capability[0]) >= 10 and triton_major <= 2:
        result = (
            False,
            "Triton 2.x does not support Blackwell; use a newer Triton/PyTorch "
            "stack or a CUDA extension",
        )
    else:
        result = True, "supported"
    _TRITON_DEVICE_SUPPORT[int(device_index)] = result
    return result


if _HAS_TRITON:

    @triton.jit
    def _triton_max_first_combine(left_value, left_index, right_value, right_index):
        """Reduce by maximum value, breaking exact ties toward the first index."""
        take_left = (left_value > right_value) | (
            (left_value == right_value) & (left_index < right_index)
        )
        return (
            tl.where(take_left, left_value, right_value),
            tl.where(take_left, left_index, right_index),
        )


    @triton.jit
    def _sift_vla_greedy_persistent_kernel(
        relevance_scores_ptr,
        pairwise_distances_ptr,
        selection_order_ptr,
        token_budget,
        pairwise_distance_stride,
        BLOCK_N: tl.constexpr,
    ):

        offsets = tl.arange(0, BLOCK_N)
        relevance_scores = tl.load(relevance_scores_ptr + offsets).to(tl.float32)

        _, first_idx = tl.reduce(
            (relevance_scores, offsets),
            axis=0,
            combine_fn=_triton_max_first_combine,
        )
        tl.store(selection_order_ptr, first_idx)

        selected = offsets == first_idx
        min_distances = tl.load(
            pairwise_distances_ptr + first_idx * pairwise_distance_stride + offsets
        ).to(tl.float32)
        neg_inf = -3.4028234663852886e38

        for step in range(1, token_budget):
            distance_min = tl.min(min_distances, axis=0)
            distance_max = tl.max(min_distances, axis=0)
            distance_denom = distance_max - distance_min + 1.0e-6
            diversity_scores = (min_distances - distance_min) / distance_denom

            relevance_values = tl.where(selected, neg_inf, relevance_scores)
            diversity_values = tl.where(selected, neg_inf, diversity_scores)

            relevance_best, relevance_candidate = tl.reduce(
                (relevance_values, offsets),
                axis=0,
                combine_fn=_triton_max_first_combine,
            )
            diversity_best, diversity_candidate = tl.reduce(
                (diversity_values, offsets),
                axis=0,
                combine_fn=_triton_max_first_combine,
            )

            relevance_at_diversity_candidate = tl.load(relevance_scores_ptr + diversity_candidate).to(tl.float32)
            # One-hot reduction supports Triton versions without dynamic block indexing.
            diversity_at_relevance_candidate = tl.sum(
                tl.where(offsets == relevance_candidate, diversity_scores, 0.0),
                axis=0,
            )
            relevance_gap = relevance_best - relevance_at_diversity_candidate
            diversity_gap = diversity_best - diversity_at_relevance_candidate
            zero_gap = (relevance_gap + diversity_gap) == 0.0
            joint_scores = relevance_gap * relevance_scores + diversity_gap * diversity_scores
            joint_scores = tl.where(selected, neg_inf, joint_scores)
            _, joint_candidate = tl.reduce(
                (joint_scores, offsets),
                axis=0,
                combine_fn=_triton_max_first_combine,
            )
            next_idx = tl.where(zero_gap, relevance_candidate, joint_candidate)
            tl.store(selection_order_ptr + step, next_idx)

            selected = selected | (offsets == next_idx)
            next_dist = tl.load(
                pairwise_distances_ptr + next_idx * pairwise_distance_stride + offsets
            ).to(tl.float32)
            min_distances = tl.minimum(min_distances, next_dist)


def _skip_result(hidden_states: torch.Tensor) -> SIFTVLAResult:
    keep_indices = torch.arange(hidden_states.shape[1], device=hidden_states.device, dtype=torch.long)
    return SIFTVLAResult(hidden_states=hidden_states, keep_indices=keep_indices, applied=False)


def _valid_text_mask(
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
    vision_end: int,
    text_len: int,
    device: torch.device,
) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones((batch_size, text_len), dtype=torch.bool, device=device)
    if attention_mask.shape != (batch_size, seq_len):
        raise ValueError(f"SIFT-VLA attention_mask must have shape {(batch_size, seq_len)}.")
    return attention_mask[:, vision_end:].bool()


def _instruction_text_mask(text_mask: torch.Tensor, text_len: int, cfg: Any) -> torch.Tensor:
    """Limit valid text states to the instruction token span."""
    start = getattr(cfg, "sift_vla_instruction_text_start", None)
    end = getattr(cfg, "sift_vla_instruction_text_end", None)
    if start is None and end is None:
        return text_mask
    if start is None or end is None:
        raise ValueError("SIFT-VLA instruction text start and end must be set together.")
    if start < 0 or end <= start or end > text_len:
        raise ValueError(
            "Invalid instruction text span for SIFT-VLA: "
            f"start={start}, end={end}, text_len={text_len}"
        )

    instruction_mask = torch.zeros_like(text_mask)
    instruction_mask[:, start:end] = True
    return text_mask & instruction_mask


def _normalize_01(x: torch.Tensor) -> torch.Tensor:
    x_min, x_max = torch.aminmax(x)
    return (x - x_min) / (x_max - x_min + 1e-6)


def _select_eager_from_prepared(
    relevance_scores: torch.Tensor,
    pairwise_distances: torch.Tensor,
    token_budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Algorithm 1, normalizing diversity over all N visual tokens.

    Ties use the first index. Zero gaps select i_R.
    """
    device = relevance_scores.device
    num_visual_tokens = int(relevance_scores.shape[0])
    selection_order = torch.empty((token_budget,), device=device, dtype=torch.long)
    first_idx = torch.argmax(relevance_scores)
    selection_order[0] = first_idx

    selected = torch.zeros((num_visual_tokens,), device=device, dtype=torch.bool)
    selected.scatter_(0, first_idx.reshape(1), True)
    min_distances = pairwise_distances.index_select(0, first_idx.reshape(1)).squeeze(0).clone()
    neg_inf = torch.finfo(relevance_scores.dtype).min

    for step in range(1, token_budget):
        diversity_scores = _normalize_01(min_distances)
        candidate_scores = torch.stack((relevance_scores, diversity_scores), dim=0)
        candidate_scores.masked_fill_(selected.unsqueeze(0), neg_inf)
        candidate_indices = torch.argmax(candidate_scores, dim=1)
        relevance_candidate = candidate_indices[0]
        diversity_candidate = candidate_indices[1]
        relevance_gap = relevance_scores[relevance_candidate] - relevance_scores[diversity_candidate]
        diversity_gap = diversity_scores[diversity_candidate] - diversity_scores[relevance_candidate]
        zero_gap = (relevance_gap + diversity_gap).eq(0.0)
        joint_scores = relevance_gap * relevance_scores + diversity_gap * diversity_scores
        joint_candidate = torch.argmax(joint_scores.masked_fill(selected, neg_inf))
        next_idx = torch.where(zero_gap, relevance_candidate, joint_candidate)
        selection_order[step] = next_idx

        # The final selection does not need another distance-row update.
        if step + 1 < token_budget:
            selected.scatter_(0, next_idx.reshape(1), True)
            next_dist = pairwise_distances.index_select(0, next_idx.reshape(1)).squeeze(0)
            min_distances = torch.minimum(min_distances, next_dist)

    return selection_order.sort().values, selection_order


def _select_triton_from_prepared(
    relevance_scores: torch.Tensor,
    pairwise_distances: torch.Tensor,
    token_budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the N=256 persistent recurrence kernel exactly once."""
    if not _HAS_TRITON:
        raise RuntimeError(f"Triton is unavailable: {_TRITON_IMPORT_ERROR}")
    if not relevance_scores.is_cuda or not pairwise_distances.is_cuda:
        raise ValueError("The Triton selector requires CUDA tensors")
    if int(relevance_scores.numel()) != 256 or tuple(pairwise_distances.shape) != (256, 256):
        raise ValueError("The persistent Triton selector currently requires N=256")

    relevance_scores = relevance_scores.float().contiguous()
    pairwise_distances = pairwise_distances.float().contiguous()
    selection_order = torch.empty(
        (token_budget,),
        device=relevance_scores.device,
        dtype=torch.long,
    )
    _sift_vla_greedy_persistent_kernel[(1,)](
        relevance_scores,
        pairwise_distances,
        selection_order,
        int(token_budget),
        pairwise_distances.stride(0),
        BLOCK_N=256,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    selected_indices = selection_order.sort().values
    return selected_indices, selection_order


def _select_visual_tokens(
    relevance_scores: torch.Tensor,
    normalized_visual_features: torch.Tensor,
    token_budget: int,
    *,
    backend: str = "eager",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select using the normalized relevance R from Eq. 2 and unit visual states."""
    num_visual_tokens = int(relevance_scores.shape[0])
    token_budget = max(1, min(token_budget, num_visual_tokens))
    backend = backend.strip().lower()
    if backend not in {"eager", "triton", "auto"}:
        raise ValueError(
            "sift_vla_selector_backend must be one of: eager, triton, auto; "
            f"got {backend!r}"
        )

    # Both backends consume the same float32 scores and distances.
    visual_features_float = normalized_visual_features.float()
    pairwise_distances = 1.0 - torch.matmul(visual_features_float, visual_features_float.transpose(0, 1))
    pairwise_distances.clamp_min_(0.0)
    pairwise_distances = pairwise_distances.float().contiguous()
    relevance_scores = relevance_scores.float().contiguous()

    eager_args = (relevance_scores, pairwise_distances, token_budget)
    if backend == "eager":
        return _select_eager_from_prepared(*eager_args)

    triton_device_ok, triton_device_reason = _triton_device_supported(relevance_scores)
    triton_eligible = bool(
        _HAS_TRITON
        and relevance_scores.is_cuda
        and triton_device_ok
        and num_visual_tokens == 256
        and tuple(pairwise_distances.shape) == (256, 256)
    )
    if not triton_eligible:
        if not _HAS_TRITON:
            reason = "Triton import failed"
        elif not triton_device_ok:
            reason = triton_device_reason
        else:
            reason = "the persistent backend requires CUDA and N=256"
        if backend == "triton":
            raise RuntimeError(f"Cannot use sift_vla_selector_backend='triton': {reason}")
        return _select_eager_from_prepared(*eager_args)

    return _select_triton_from_prepared(relevance_scores, pairwise_distances, token_budget)


def sift_vla_prune(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    cfg: Optional[Any] = None,
) -> SIFTVLAResult:
    """Prune [prefix/BOS] + [visual tokens] + [text/prompt tokens].

    Instruction spans are relative to the text suffix and contain at least
    one valid instruction token, as guaranteed by evaluation. Retained tokens keep
    their original order. The OpenVLA pruning hooks support one example at a
    time, so selection never mixes observations or instructions across a batch.
    """
    batch_size, sequence_length, _ = hidden_states.shape
    device = hidden_states.device
    vision_start = int(getattr(cfg, "sift_vla_vision_start", 1))
    num_visual_tokens = int(getattr(cfg, "sift_vla_num_visual_tokens", 256))
    token_budget = int(getattr(cfg, "sift_vla_token_budget", 64))
    if vision_start < 0 or num_visual_tokens < 1:
        raise ValueError("SIFT-VLA requires vision_start >= 0 and num_visual_tokens >= 1")

    configured_backend = getattr(cfg, "sift_vla_selector_backend", None)
    if configured_backend is None:
        configured_backend = os.environ.get("SIFT_VLA_SELECTOR_BACKEND", "auto")
    selector_backend = configured_backend.strip().lower()
    vision_end = vision_start + num_visual_tokens
    token_budget = max(1, min(token_budget, num_visual_tokens))
    if token_budget >= num_visual_tokens:
        return _skip_result(hidden_states)
    if sequence_length <= vision_end:
        return _skip_result(hidden_states)
    if batch_size != 1:
        raise ValueError("SIFT-VLA pruning requires batch_size=1; run each observation separately.")

    visual_hidden_states = hidden_states[:, vision_start:vision_end, :]
    text_hidden_states = hidden_states[:, vision_end:, :]
    text_length = int(text_hidden_states.shape[1])
    instruction_mask = _valid_text_mask(
        attention_mask, batch_size, sequence_length, vision_end, text_length, device,
    )
    instruction_mask = _instruction_text_mask(instruction_mask, text_length, cfg)
    valid_instruction_counts = instruction_mask.sum(dim=1)

    # Eq. 1: mean instruction hidden state q.
    instruction_representation = text_hidden_states.float().masked_fill(
        ~instruction_mask.unsqueeze(-1), 0.0,
    ).sum(dim=1) / valid_instruction_counts.unsqueeze(1)

    # Eq. 2: cosine relevance R, normalized independently for each example.
    normalized_visual_features = F.normalize(visual_hidden_states.float(), p=2, dim=-1)
    normalized_instruction = F.normalize(instruction_representation, p=2, dim=-1)
    cosine_relevance = (normalized_visual_features * normalized_instruction.unsqueeze(1)).sum(dim=-1)
    relevance_scores = _normalize_01(cosine_relevance)
    selection_relevance = relevance_scores[0]
    selection_visual_features = normalized_visual_features[0]
    selected_indices, _ = _select_visual_tokens(
        selection_relevance, selection_visual_features, token_budget, backend=selector_backend,
    )

    keep_prefix = torch.arange(vision_start, device=device, dtype=torch.long)
    keep_visual = selected_indices + vision_start
    keep_suffix = torch.arange(vision_end, sequence_length, device=device, dtype=torch.long)
    keep_indices = torch.cat([keep_prefix, keep_visual, keep_suffix], dim=0)
    pruned_hidden_states = hidden_states[:, keep_indices, :]
    return SIFTVLAResult(hidden_states=pruned_hidden_states, keep_indices=keep_indices)
