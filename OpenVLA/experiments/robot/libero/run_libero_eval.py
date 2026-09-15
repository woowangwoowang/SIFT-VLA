"""Evaluate an OpenVLA checkpoint on LIBERO; see run_eval/ for suite launchers."""

import os
import re
from contextlib import closing
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from rd_token.config import SIFTVLAConfig


def _safe_slug(value) -> str:
    """Make a compact filesystem-safe tag for experiment log filenames."""
    text = str(value)
    text = text.replace("/", "_")
    text = text.replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_+=.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "na"


def make_sift_vla_experiment_tag(cfg) -> str:
    """Encode the SIFT-VLA layer and retained visual-token budget in log names."""
    if not cfg.sift_vla_enable:
        return "sift_vla_off"

    layer = _safe_slug(cfg.sift_vla_layer)
    token_budget = _safe_slug(cfg.sift_vla_token_budget)
    cache = "cache_on" if cfg.model_use_cache else "cache_off"
    return f"sift_vla_L{layer}_K{token_budget}_{cache}"


def _config_lines(cfg, experiment_tag: str, local_log_filepath: str) -> list[str]:
    """Log every CLI setting once, including inherited SIFT-VLA settings."""
    lines = [
        "=" * 80,
        "EXPERIMENT CONFIG",
        f"experiment_tag={experiment_tag}",
        f"local_log_filepath={local_log_filepath}",
        f"unnorm_key={cfg.unnorm_key}",
    ]
    lines.extend(f"{field.name}={getattr(cfg, field.name)}" for field in fields(cfg))
    lines.append("=" * 80)
    return lines


def log_runtime_config(log_file, cfg, experiment_tag: str, local_log_filepath: str) -> None:
    lines = _config_lines(cfg, experiment_tag, local_log_filepath)
    for line in lines:
        print(line)
        log_file.write(line + "\n")
    log_file.flush()


@dataclass
class GenerateConfig(SIFTVLAConfig):
    model_use_cache: bool = True

    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True  # Match checkpoints trained with image augmentation.

    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10  # Let dropped objects settle before the first action.
    num_trials_per_task: int = 50

    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_project: str = "YOUR_WANDB_PROJECT"
    wandb_entity: str = "YOUR_WANDB_ENTITY"

    seed: int = 7


# Maximum rollout lengths follow the longest demonstrations in each suite.
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def validate_eval_config(cfg: GenerateConfig) -> None:
    """Reject invalid evaluation settings before loading a checkpoint."""
    if not cfg.pretrained_checkpoint:
        raise ValueError("--pretrained_checkpoint is required.")
    if cfg.model_family != "openvla":
        raise ValueError("--model_family must be openvla.")
    if cfg.task_suite_name not in MAX_STEPS:
        raise ValueError(f"Unknown LIBERO suite: {cfg.task_suite_name}.")
    if cfg.num_trials_per_task < 1:
        raise ValueError("--num_trials_per_task must be positive.")
    if cfg.num_steps_wait < 0:
        raise ValueError("--num_steps_wait must be nonnegative.")
    if not 0 <= cfg.seed <= 2**32 - 1:
        raise ValueError("--seed must be between 0 and 4294967295.")
    if "image_aug" in str(cfg.pretrained_checkpoint) and not cfg.center_crop:
        raise ValueError("Image-augmented checkpoints require center_crop=True.")
    if cfg.load_in_8bit and cfg.load_in_4bit:
        raise ValueError("Cannot use both 8-bit and 4-bit quantization.")


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    validate_eval_config(cfg)
    max_steps = MAX_STEPS[cfg.task_suite_name]

    if cfg.use_wandb:
        import wandb

    set_seed_everywhere(cfg.seed)

    cfg.unnorm_key = cfg.task_suite_name

    experiment_tag = make_sift_vla_experiment_tag(cfg)

    model = get_model(cfg)

    if cfg.model_family == "openvla":
        # Some fine-tuned checkpoints use the _no_noops dataset suffix.
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}--{experiment_tag}--{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{_safe_slug(cfg.run_id_note)}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    with open(local_log_filepath, "w") as log_file:
        print(f"Logging to local log file: {local_log_filepath}")
        log_runtime_config(log_file, cfg, experiment_tag, local_log_filepath)

        if cfg.use_wandb:
            wandb.init(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                name=run_id,
            )

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[cfg.task_suite_name]()
        num_tasks_in_suite = task_suite.n_tasks
        print(f"Task suite: {cfg.task_suite_name}")
        log_file.write(f"Task suite: {cfg.task_suite_name}\n")

        resize_size = get_image_resize_size(cfg)

        total_episodes, total_successes = 0, 0
        for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
            task = task_suite.get_task(task_id)

            initial_states = task_suite.get_task_init_states(task_id)
            if cfg.num_trials_per_task > len(initial_states):
                raise ValueError(
                    f"Task {task_id} has {len(initial_states)} initial states, "
                    f"but {cfg.num_trials_per_task} trials were requested."
                )

            env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

            with closing(env):
                task_episodes, task_successes = 0, 0
                for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
                    print(f"\nTask: {task_description}")
                    log_file.write(f"\nTask: {task_description}\n")

                    env.reset()

                    obs = env.set_init_state(initial_states[episode_idx])

                    t = 0
                    replay_images = []
                    done = False

                    print(f"Starting episode {task_episodes+1}...")
                    log_file.write(f"Starting episode {task_episodes+1}...\n")
                    while t < max_steps + cfg.num_steps_wait:
                        # Let objects settle before querying the policy.
                        if t < cfg.num_steps_wait:
                            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                            t += 1
                            continue

                        img = get_libero_image(obs, resize_size)

                        replay_images.append(img)

                        observation = {
                            "full_image": img,
                            "state": np.concatenate(
                                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            ),
                        }

                        action = get_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                        )

                        # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                        action = normalize_gripper_action(action, binarize=True)

                        # Convert the training gripper convention to LIBERO: -1=open, +1=close.
                        if cfg.model_family == "openvla":
                            action = invert_gripper_action(action)

                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    task_episodes += 1
                    total_episodes += 1

                    save_rollout_video(
                        replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
                    )

                    print(f"Success: {done}")
                    print(f"# episodes completed so far: {total_episodes}")
                    print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
                    log_file.write(f"Success: {done}\n")
                    log_file.write(f"# episodes completed so far: {total_episodes}\n")
                    log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
                    log_file.flush()

                print(f"Current task success rate: {task_successes / task_episodes}")
                print(f"Current total success rate: {total_successes / total_episodes}")
                log_file.write(f"Current task success rate: {task_successes / task_episodes}\n")
                log_file.write(f"Current total success rate: {total_successes / total_episodes}\n")
                log_file.flush()

                if cfg.use_wandb:
                    wandb.log(
                        {
                            f"success_rate/{task_description}": task_successes / task_episodes,
                            f"num_episodes/{task_description}": task_episodes,
                        }
                    )

    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": total_successes / total_episodes,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
