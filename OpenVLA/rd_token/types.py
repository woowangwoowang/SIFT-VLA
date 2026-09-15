"""SIFT-VLA selection result shared by the pruning hooks."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SIFTVLAResult:
    hidden_states: torch.Tensor
    # Original sequence indices, including unchanged nonvisual tokens.
    keep_indices: Optional[torch.Tensor] = None
    applied: bool = True
