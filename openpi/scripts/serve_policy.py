import dataclasses
import enum
import logging
from typing import Literal

import torch
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Seed PyTorch policy sampling, including flow-matching noise.
    seed: int = 7
    # Record the policy's behavior for debugging.
    record: bool = False

    # Enable SIFT-VLA for a PyTorch pi0.5 checkpoint.
    sift_vla_enable: bool = False
    # Zero-based decoder layer to prune before.
    sift_vla_layer: int = 3
    # Total retained visual tokens across valid camera views.
    sift_vla_token_budget: int = 128
    # Backend for token selection.
    sift_vla_selector_backend: Literal["auto", "eager", "triton"] = "auto"
    # Reuse the prefix KV cache during denoising.
    model_use_cache: bool = True

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def _sample_kwargs(args: Args) -> dict:
    if not args.sift_vla_enable:
        return {"sift_vla_enable": False, "model_use_cache": args.model_use_cache}
    return {
        field.name: getattr(args, field.name)
        for field in dataclasses.fields(args)
        if field.name.startswith("sift_vla_") or field.name == "model_use_cache"
    }


def create_default_policy(args: Args, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    env = args.env
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            sample_kwargs=_sample_kwargs(args),
            default_prompt=default_prompt,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                sample_kwargs=_sample_kwargs(args),
                default_prompt=args.default_prompt,
            )
        case Default():
            return create_default_policy(args, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535.")
    if args.sift_vla_enable:
        if args.sift_vla_layer < 0 or args.sift_vla_token_budget < 1:
            raise ValueError("SIFT-VLA requires a nonnegative layer and a positive token budget.")
        if args.sift_vla_selector_backend not in {"auto", "eager", "triton"}:
            raise ValueError("--sift-vla-selector-backend must be auto, eager, or triton.")
    if not 0 <= args.seed <= 2**32 - 1:
        raise ValueError("--seed must be between 0 and 4294967295.")
    torch.manual_seed(args.seed)
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    logging.info("Serving policy on 0.0.0.0:%s", args.port)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
