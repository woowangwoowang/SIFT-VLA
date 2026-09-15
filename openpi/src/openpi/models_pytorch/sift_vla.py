"""SIFT-VLA visual-token selection (paper Eqs. 1--8 and Algorithm 1).

All valid camera tokens form one candidate pool with total retained budget K.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
    _TRITON_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    triton = None
    tl = None
    _HAS_TRITON = False
    _TRITON_IMPORT_ERROR = exc


_TRITON_DEVICE_SUPPORT: dict[int, tuple[bool, str]] = {}


def _triton_device_supported(tensor: torch.Tensor) -> tuple[bool, str]:
    if not _HAS_TRITON or not tensor.is_cuda:
        return False, "Triton and CUDA are required"

    device_index = tensor.device.index
    if device_index is None:
        device_index = int(torch.cuda.current_device())
    cached = _TRITON_DEVICE_SUPPORT.get(int(device_index))
    if cached is not None:
        return cached

    capability = torch.cuda.get_device_capability(device_index)
    try:
        triton_major = int(str(getattr(triton, "__version__", "0")).split(".", 1)[0])
    except (TypeError, ValueError):
        triton_major = 0

    if int(capability[0]) >= 10 and triton_major <= 2:
        result = (
            False,
            "Triton 2.x does not support Blackwell; use a newer Triton/PyTorch stack or a CUDA extension",
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
            diversity_scores = tl.div_rn(min_distances - distance_min, distance_denom)

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


@dataclass
class SIFTVLAResult:
    prefix_hidden_states: torch.Tensor
    prefix_keep_indices: torch.Tensor
    applied: bool


def should_apply(config: dict[str, Any] | None, layer_idx: int) -> bool:
    return bool(config and config.get("enabled", False)) and int(layer_idx) == int(config.get("layer", 3))


def _normalize_01(scores: torch.Tensor) -> torch.Tensor:
    minimum, maximum = torch.aminmax(scores)
    return (scores - minimum) / (maximum - minimum + 1e-6)


def _select_eager_from_prepared(
    relevance_scores: torch.Tensor,
    pairwise_distances: torch.Tensor,
    token_budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Algorithm 1; normalize diversity over all N visual positions."""
    device = relevance_scores.device
    selection_order = torch.empty(token_budget, device=device, dtype=torch.long)
    first_idx = torch.argmax(relevance_scores)
    selection_order[0] = first_idx
    selected = torch.zeros_like(relevance_scores, dtype=torch.bool)
    selected.scatter_(0, first_idx.reshape(1), True)  # noqa: FBT003
    min_distances = pairwise_distances.index_select(0, first_idx.reshape(1)).squeeze(0)
    neg_inf = torch.finfo(relevance_scores.dtype).min

    for step in range(1, token_budget):
        diversity_scores = _normalize_01(min_distances)
        candidates = torch.stack((relevance_scores, diversity_scores))
        candidates.masked_fill_(selected.unsqueeze(0), neg_inf)
        relevance_candidate, diversity_candidate = candidates.argmax(dim=1).unbind()
        relevance_gap = relevance_scores[relevance_candidate] - relevance_scores[diversity_candidate]
        diversity_gap = diversity_scores[diversity_candidate] - diversity_scores[relevance_candidate]
        joint_scores = relevance_gap * relevance_scores + diversity_gap * diversity_scores
        joint_candidate = joint_scores.masked_fill(selected, neg_inf).argmax()
        next_idx = torch.where((relevance_gap + diversity_gap).eq(0), relevance_candidate, joint_candidate)
        selection_order[step] = next_idx
        if step + 1 < token_budget:
            selected.scatter_(0, next_idx.reshape(1), True)  # noqa: FBT003
            next_distances = pairwise_distances.index_select(0, next_idx.reshape(1)).squeeze(0)
            min_distances = torch.minimum(min_distances, next_distances)
    return selection_order.sort().values, selection_order


def _select_triton_from_prepared(
    relevance_scores: torch.Tensor,
    pairwise_distances: torch.Tensor,
    token_budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _HAS_TRITON:
        raise RuntimeError(f"Triton is unavailable: {_TRITON_IMPORT_ERROR}")
    if not relevance_scores.is_cuda or not pairwise_distances.is_cuda:
        raise ValueError("The Triton selector requires CUDA tensors")
    num_visual_tokens = int(relevance_scores.numel())
    if num_visual_tokens not in {256, 512} or tuple(pairwise_distances.shape) != (num_visual_tokens, num_visual_tokens):
        raise ValueError("The persistent Triton selector requires N=256 or N=512")

    relevance_scores = relevance_scores.float().contiguous()
    pairwise_distances = pairwise_distances.float().contiguous()
    selection_order = torch.empty(token_budget, device=relevance_scores.device, dtype=torch.long)
    _sift_vla_greedy_persistent_kernel[(1,)](
        relevance_scores, pairwise_distances, selection_order, token_budget, pairwise_distances.stride(0),
        BLOCK_N=num_visual_tokens,
        num_warps=4 if num_visual_tokens == 256 else 8,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return selection_order.sort().values, selection_order


def _select_visual_tokens(
    relevance_scores: torch.Tensor,
    normalized_visual_features: torch.Tensor,
    token_budget: int,
    *,
    backend: str = "eager",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select from already normalized relevance R and unit visual states."""
    num_visual_tokens = int(relevance_scores.numel())
    token_budget = max(1, min(int(token_budget), num_visual_tokens))
    backend = backend.strip().lower()
    if backend not in {"eager", "triton", "auto"}:
        raise ValueError(f"selector backend must be eager, triton, or auto; got {backend!r}")

    visual_features = normalized_visual_features.float()
    pairwise_distances = (1.0 - visual_features @ visual_features.T).clamp_min_(0).contiguous()
    relevance_scores = relevance_scores.float().contiguous()
    arguments = (relevance_scores, pairwise_distances, token_budget)
    if backend == "eager":
        return _select_eager_from_prepared(*arguments)

    supported, reason = _triton_device_supported(relevance_scores)
    if not supported or num_visual_tokens not in {256, 512}:
        if backend == "triton":
            if supported:
                reason = "the persistent backend requires N=256 or N=512"
            raise RuntimeError(f"Cannot use selector backend 'triton': {reason}")
        return _select_eager_from_prepared(*arguments)

    return _select_triton_from_prepared(*arguments)


def sift_vla_prune_prefix(
    prefix_hidden_states: torch.Tensor,
    *,
    image_token_mask: torch.Tensor,
    vision_token_mask: torch.Tensor,
    text_token_mask: torch.Tensor,
    token_budget: int = 128,
    prompt_token_ids: torch.Tensor | None = None,
    text_start: int | None = None,
    exclude_text_token_ids: tuple[int, ...] = (2, 108),
    selector_backend: str = "auto",
) -> SIFTVLAResult:
    """Prune pi0.5 image tokens, preserving prompt and original prefix order.

    Joint selection uses all valid camera tokens under one total budget K.
    Evaluation supplies a single observation with at least one valid
    instruction token.
    """
    if prefix_hidden_states.ndim != 3:
        raise ValueError(f"Expected [B, S, D] prefix states, got {tuple(prefix_hidden_states.shape)}")
    batch_size, prefix_len, _ = prefix_hidden_states.shape
    if batch_size != 1:
        raise ValueError("SIFT-VLA pruning requires batch_size=1; run each observation separately.")
    device = prefix_hidden_states.device
    for name, mask in (("image_token_mask", image_token_mask), ("vision_token_mask", vision_token_mask),
                       ("text_token_mask", text_token_mask)):
        if mask.shape != (batch_size, prefix_len):
            raise ValueError(f"{name} shape {tuple(mask.shape)} does not match prefix shape")
    image_token_mask = image_token_mask.to(device=device, dtype=torch.bool)
    vision_token_mask = vision_token_mask.to(device=device, dtype=torch.bool)
    # The same mask is reused at every denoising step.
    instruction_mask = text_token_mask.to(device=device, dtype=torch.bool).clone()
    if prompt_token_ids is not None and exclude_text_token_ids:
        prompt_ids = torch.as_tensor(prompt_token_ids, device=device).reshape(1, -1)
        if text_start is None:
            text_start = int(torch.nonzero(instruction_mask[0], as_tuple=False)[0, 0])
        text_start = int(text_start)
        text_end = text_start + prompt_ids.shape[1]
        if not 0 <= text_start <= text_end <= prefix_len:
            raise ValueError("Prompt token ids must fit within the prefix text span")
        special_tokens = torch.tensor(exclude_text_token_ids, device=device, dtype=prompt_ids.dtype)
        instruction_mask[:, text_start:text_end] &= ~torch.isin(prompt_ids, special_tokens)

    vision_indices = torch.nonzero(vision_token_mask[0], as_tuple=False).flatten()
    num_visual_tokens = int(vision_indices.numel())
    token_budget = max(1, min(int(token_budget), num_visual_tokens))
    if num_visual_tokens == 0 or token_budget >= num_visual_tokens:
        return SIFTVLAResult(prefix_hidden_states, torch.arange(prefix_len, device=device), False)

    text_indices = torch.nonzero(instruction_mask[0], as_tuple=False).flatten()
    instruction_states = prefix_hidden_states[:, text_indices].float()
    instruction_representation = instruction_states.mean(dim=1)  # q, Eq. 1
    normalized_visual_features = F.normalize(prefix_hidden_states[:, vision_indices].float(), dim=-1)
    normalized_instruction = F.normalize(instruction_representation, dim=-1)
    cosine_relevance = (normalized_visual_features * normalized_instruction.unsqueeze(1)).sum(dim=-1)
    relevance_scores = _normalize_01(cosine_relevance)
    selected_indices, _ = _select_visual_tokens(
        relevance_scores[0], normalized_visual_features[0], token_budget,
        backend=selector_backend,
    )

    non_image_indices = torch.nonzero(~image_token_mask[0], as_tuple=False).flatten()
    prefix_keep_indices = torch.cat((vision_indices[selected_indices], non_image_indices)).sort().values
    pruned_prefix = prefix_hidden_states[:, prefix_keep_indices]

    return SIFTVLAResult(pruned_prefix, prefix_keep_indices, True)
