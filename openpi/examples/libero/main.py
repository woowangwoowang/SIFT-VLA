from __future__ import annotations

import collections
from contextlib import closing
import dataclasses
import datetime
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # Resolution used to render training data.
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    video_out_path: str = "data/libero/videos"
    results_out_dir: str = "data/libero/results"
    results_out_path: str = ""  # Optional explicit result filename.
    seed: int = 7


def _validate_args(args: Args) -> None:
    if args.task_suite_name not in MAX_STEPS:
        raise ValueError(f"Unknown LIBERO task suite: {args.task_suite_name}")
    if not args.host.strip() or not 1 <= args.port <= 65535:
        raise ValueError("A host and a port between 1 and 65535 are required.")
    if min(args.resize_size, args.replan_steps, args.num_trials_per_task) < 1:
        raise ValueError("resize_size, replan_steps, and num_trials_per_task must be positive.")
    if args.num_steps_wait < 0 or not 0 <= args.seed <= 2**32 - 1:
        raise ValueError("num_steps_wait must be nonnegative; seed must be between 0 and 4294967295.")


def _get_action_chunk(client, observation: dict, replan_steps: int) -> np.ndarray:
    actions = np.asarray(client.infer(observation)["actions"])
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < replan_steps:
        raise ValueError(
            f"Expected at least {replan_steps} actions with 7 coordinates; received shape {actions.shape}."
        )
    if not np.issubdtype(actions.dtype, np.number) or not np.isfinite(actions).all():
        raise ValueError("The policy returned non-finite or non-numeric actions.")
    return actions[:replan_steps]


def eval_libero(args: Args) -> None:
    _validate_args(args)
    np.random.seed(args.seed)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    logging.info("Task suite: %s", args.task_suite_name)

    video_dir = pathlib.Path(args.video_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.datetime.now(datetime.timezone.utc).astimezone()
    run_id = started_at.strftime("%Y-%m-%d_%H-%M-%S-%f")
    results_path = (
        pathlib.Path(args.results_out_path) if args.results_out_path
        else pathlib.Path(args.results_out_dir) / f"{args.task_suite_name}_{run_id}.txt"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        f"task_suite: {args.task_suite_name}\nstarted_at: {started_at.isoformat(timespec='seconds')}\n"
        f"num_trials_per_task: {args.num_trials_per_task}\nseed: {args.seed}\n"
        f"replan_steps: {args.replan_steps}\n\n",
        encoding="utf-8",
    )
    logging.info("Saving evaluation results to: %s", results_path)

    total_episodes, total_successes = 0, 0
    with closing(_websocket_client_policy.WebsocketClientPolicy(args.host, args.port)) as client:
        for task_id in tqdm.tqdm(range(task_suite.n_tasks)):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            if args.num_trials_per_task > len(initial_states):
                raise ValueError(
                    f"Task {task_id} has {len(initial_states)} initial states, "
                    f"but {args.num_trials_per_task} trials were requested."
                )
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            task_successes = 0
            with closing(env):
                for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                    logging.info("Task: %s; episode %s", task_description, episode_idx + 1)
                    env.reset()
                    obs = env.set_init_state(initial_states[episode_idx])
                    action_plan = collections.deque()
                    replay_images = []
                    done = False
                    t = 0
                    while t < MAX_STEPS[args.task_suite_name] + args.num_steps_wait:
                        # Let dropped objects settle before querying the policy.
                        if t < args.num_steps_wait:
                            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

                        # Rotate 180 degrees to match the training observations.
                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                        wrist_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                        )
                        replay_images.append(img)

                        if not action_plan:
                            element = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": np.concatenate((
                                    obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )),
                                "prompt": str(task_description),
                            }
                            action_plan.extend(_get_action_chunk(client, element, args.replan_steps))

                        obs, _, done, _ = env.step(action_plan.popleft().tolist())
                        t += 1
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break

                    total_episodes += 1
                    suffix = "success" if done else "failure"
                    # Include run, task, and episode IDs so repeated trials retain their videos.
                    video_path = video_dir / f"{run_id}_task-{task_id}_episode-{episode_idx + 1}_{suffix}.mp4"
                    imageio.mimwrite(video_path, [np.asarray(image) for image in replay_images], fps=10)
                    logging.info("Success: %s; total: %s/%s", done, total_successes, total_episodes)
                    _append_result(results_path, [
                        f"episode task_id={task_id} episode_idx={episode_idx} success={done} steps={t}",
                        f"task: {task_description}", f"video: {video_path}",
                        f"total_episodes: {total_episodes}", f"total_successes: {total_successes}",
                        f"total_success_rate: {total_successes / total_episodes:.6f}", "",
                    ])

            task_success_rate = task_successes / args.num_trials_per_task
            logging.info("Task success rate: %.4f", task_success_rate)
            _append_result(results_path, [
                f"task_summary task_id={task_id}", f"task: {task_description}",
                f"task_episodes: {args.num_trials_per_task}", f"task_successes: {task_successes}",
                f"task_success_rate: {task_success_rate:.6f}", "",
            ])

    logging.info("Total successes: %s/%s", total_successes, total_episodes)
    _append_result(results_path, [
        "final_summary", f"total_episodes: {total_episodes}", f"total_successes: {total_successes}",
        f"total_success_rate: {total_successes / total_episodes:.6f}", "",
    ])


def _append_result(path: pathlib.Path, lines: list[str]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write("\n".join(lines) + "\n")


def _get_libero_env(task, resolution, seed):
    """Initialize a LIBERO environment and return its task description."""
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env, task.language


def _quat2axisangle(quat):
    """Convert LIBERO's (x, y, z, w) quaternion to axis-angle."""
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
