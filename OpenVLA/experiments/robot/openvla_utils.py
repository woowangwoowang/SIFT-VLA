"""Utils for evaluating the OpenVLA policy."""

import json
import os

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from rd_token.config import SIFT_VLA_KEYS


DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def _runtime_config_targets(vla):
    lm = vla.language_model
    targets = (vla.config, vla.generation_config, lm.config, lm.generation_config, lm.model.config)
    return [target for target in targets if target is not None]


def _set_instruction_text_span(vla, processor, prompt, instruction, instruction_char_start):
    """Map instruction characters to text_h, where text_h[0] is input_ids[1]."""
    tokenized = processor.tokenizer(
        prompt,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offsets = tokenized["offset_mapping"]
    instruction_char_end = instruction_char_start + len(instruction)
    instruction_token_indices = [
        idx
        for idx, (start, end) in enumerate(offsets)
        if end > start and end > instruction_char_start and start < instruction_char_end
    ]
    if not instruction_token_indices:
        raise ValueError("Could not locate instruction tokens in the OpenVLA prompt.")

    # Convert input_ids indices to end-exclusive indices into text_h.
    text_start = instruction_token_indices[0] - 1
    text_end = instruction_token_indices[-1]
    if text_start < 0:
        raise ValueError("Instruction unexpectedly overlaps the BOS token.")

    for target in _runtime_config_targets(vla):
        target.sift_vla_instruction_text_start = text_start
        target.sift_vla_instruction_text_end = text_end


def propagate_runtime_config_to_vla(vla, cfg):
    """Copy evaluation settings into OpenVLA and LLaMA configs."""
    runtime_keys = [*SIFT_VLA_KEYS, "model_use_cache"]
    targets = _runtime_config_targets(vla)
    for key in runtime_keys:
        value = getattr(cfg, key)
        for target in targets:
            setattr(target, key, value)
    for target in targets:
        target.use_cache = cfg.model_use_cache


def _register_openvla_auto_classes():
    # Register OpenVLA model to HF Auto Classes so local prismatic code is used.
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    if cfg.sift_vla_enable:
        from transformers.models.llama import modeling_llama

        if not callable(getattr(modeling_llama, "apply_sift_vla_if_needed", None)):
            raise RuntimeError(
                "SIFT-VLA requires this repository's modified Transformers. "
                "From the OpenVLA directory, run `python -m pip install -e ./transformers` "
                "or use the run_eval/ launchers to select the bundled code via PYTHONPATH."
            )
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    _register_openvla_auto_classes()

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # A local statistics file overrides the values stored in the checkpoint.
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats

    propagate_runtime_config_to_vla(vla, cfg)

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    _register_openvla_auto_classes()
    return AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=False)


def crop_and_resize(image, crop_scale, batch_size):
    """Center-crop by area and resize to 224 x 224, matching dlimp preprocessing."""
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(
    cfg,
    vla,
    processor,
    base_vla_name,
    obs,
    task_label,
    unnorm_key,
    center_crop=False,
):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = crop_and_resize(image, crop_scale, batch_size)
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    instruction = task_label.lower()
    if "openvla-v01" in str(base_vla_name):
        prompt_prefix = f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to "
        prompt = f"{prompt_prefix}{instruction}? ASSISTANT:"
    else:
        prompt_prefix = "In: What action should the robot take to "
        prompt = f"{prompt_prefix}{instruction}?\nOut:"

    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    if cfg.sift_vla_enable:
        _set_instruction_text_span(vla, processor, prompt, instruction, len(prompt_prefix))

    return vla.predict_action(
        **inputs,
        unnorm_key=unnorm_key,
        do_sample=False,
        use_cache=cfg.model_use_cache,
    )
