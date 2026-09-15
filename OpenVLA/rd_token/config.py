"""SIFT-VLA inference settings."""

from dataclasses import dataclass, fields


@dataclass
class SIFTVLAConfig:
    sift_vla_enable: bool = False

    # Zero-based index 3: process layers 0, 1, 2, then prune before layer 3.
    sift_vla_layer: int = 3
    sift_vla_token_budget: int = 64
    sift_vla_vision_start: int = 1
    sift_vla_num_visual_tokens: int = 256

    sift_vla_selector_backend: str = "auto"


SIFT_VLA_KEYS = [field.name for field in fields(SIFTVLAConfig)]
