import dataclasses
from typing import Literal

import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

from openpi.models_pytorch import sift_vla
from openpi.models_pytorch.attention import attention_forward


@dataclasses.dataclass
class LayerwisePrefixKVCache:
    """Prefix-only KV cache whose physical sequence length may vary by layer."""

    key_cache: list[torch.Tensor | None]
    value_cache: list[torch.Tensor | None]
    prefix_pad_masks: list[torch.Tensor | None]

    @classmethod
    def empty(cls, num_layers: int) -> "LayerwisePrefixKVCache":
        return cls(
            key_cache=[None] * num_layers,
            value_cache=[None] * num_layers,
            prefix_pad_masks=[None] * num_layers,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        _cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Implement the small Cache interface used by GemmaAttention."""
        if self.key_cache[layer_idx] is not None:
            raise RuntimeError(f"Prefix KV cache layer {layer_idx} was populated more than once")
        self.key_cache[layer_idx] = key_states
        self.value_cache[layer_idx] = value_states
        return key_states, value_states

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        key_states = self.key_cache[layer_idx]
        value_states = self.value_cache[layer_idx]
        if key_states is None or value_states is None:
            raise RuntimeError(f"Prefix KV cache layer {layer_idx} has not been populated")
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        key_states = self.key_cache[layer_idx]
        return 0 if key_states is None else int(key_states.shape[-2])

    def pad_mask(self, layer_idx: int) -> torch.Tensor:
        mask = self.prefix_pad_masks[layer_idx]
        if mask is None:
            raise RuntimeError(f"Prefix pad mask for layer {layer_idx} has not been populated")
        return mask


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        if getattr(modeling_gemma, "OPENPI_BLOCK_ATTENTION_VERSION", 0) < 1:
            raise RuntimeError(
                "The installed Transformers Gemma patch is outdated. Copy this repository's "
                "src/openpi/models_pytorch/transformers_replace/ contents into the active environment's "
                "transformers package, following the PyTorch installation step in README.md. "
                "Both cached and joint attention paths require the updated block-attention patch."
            )

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None
        self.to_bfloat16_for_selected_params(precision)

    def build_sift_vla_prefix_cache(
        self,
        prefix_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        prefix_pad_masks: torch.Tensor,
        sift_vla_config: dict,
    ) -> LayerwisePrefixKVCache:
        """Prefill a layerwise prefix cache, pruning once at the configured layer."""
        model = self.paligemma.language_model
        num_layers = int(model.config.num_hidden_layers)
        hidden_states = prefix_embeds
        current_attention_mask = attention_mask
        current_position_ids = position_ids
        current_pad_masks = prefix_pad_masks
        cache = LayerwisePrefixKVCache.empty(num_layers)

        for layer_idx, layer in enumerate(model.layers[:num_layers]):
            if sift_vla.should_apply(sift_vla_config, layer_idx):
                result = sift_vla.sift_vla_prune_prefix(
                    hidden_states,
                    image_token_mask=sift_vla_config["prefix_image_token_mask"],
                    vision_token_mask=sift_vla_config["prefix_vision_token_mask"],
                    text_token_mask=sift_vla_config["prefix_text_token_mask"],
                    token_budget=int(sift_vla_config.get("token_budget", 128)),
                    prompt_token_ids=sift_vla_config.get("prompt_token_ids"),
                    text_start=sift_vla_config.get("text_start"),
                    selector_backend=str(sift_vla_config.get("selector_backend", "auto")),
                )
                if result.applied:
                    keep_indices = result.prefix_keep_indices
                    hidden_states = result.prefix_hidden_states
                    current_attention_mask = current_attention_mask.index_select(-2, keep_indices).index_select(
                        -1, keep_indices
                    )
                    current_position_ids = current_position_ids.index_select(1, keep_indices)
                    current_pad_masks = current_pad_masks.index_select(1, keep_indices)

            if layer.self_attn.q_proj.weight.dtype == torch.bfloat16:
                hidden_states = hidden_states.to(dtype=torch.bfloat16)
            position_embeddings = model.rotary_emb(hidden_states, current_position_ids)
            layer_outputs = layer(
                hidden_states,
                attention_mask=current_attention_mask,
                position_ids=current_position_ids,
                past_key_value=cache,
                output_attentions=False,
                use_cache=True,
                # Position embeddings are already built from
                # current_position_ids.  The layerwise cache does not use a
                # cache_position, so avoid launching an arange kernel for
                # every decoder layer.
                cache_position=None,
                position_embeddings=position_embeddings,
                adarms_cond=None,
            )
            hidden_states = layer_outputs[0]
            cache.prefix_pad_masks[layer_idx] = current_pad_masks

        return cache

    def forward_expert_with_layerwise_prefix_cache(
        self,
        suffix_embeds: torch.Tensor,
        attention_masks: list[torch.Tensor],
        position_ids: torch.LongTensor,
        prefix_cache: LayerwisePrefixKVCache,
        adarms_cond: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run only the action expert while reading layer-specific prefix KVs."""
        model = self.gemma_expert.model
        num_layers = int(model.config.num_hidden_layers)
        if len(attention_masks) != num_layers or len(prefix_cache.key_cache) != num_layers:
            raise ValueError(
                "Layerwise cache/mask depth does not match the action expert: "
                f"cache={len(prefix_cache.key_cache)}, masks={len(attention_masks)}, model={num_layers}"
            )

        hidden_states = suffix_embeds
        if model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype=torch.bfloat16)

        # The no-cache joint path uses the PaliGemma rotary embedding for both
        # streams, so use it here as well to preserve identical RoPE values.
        position_embeddings = self.paligemma.language_model.rotary_emb(hidden_states, position_ids)
        for layer_idx, layer in enumerate(model.layers[:num_layers]):
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_masks[layer_idx],
                position_ids=position_ids,
                past_key_value=prefix_cache,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
                adarms_cond=adarms_cond,
            )
            hidden_states = layer_outputs[0]

        hidden_states, _ = model.norm(hidden_states, cond=adarms_cond)
        return hidden_states

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
        sift_vla_config: dict | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers
            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # Attention computation
                att_output, _ = attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                    dropout=(
                        self.paligemma.language_model.layers[layer_idx].self_attn.attention_dropout
                        if self.training else 0.0
                    ),
                )
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, query_states.shape[1] * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            def apply_sift_vla_if_needed(layer_idx, inputs_embeds, attention_mask, position_ids):
                if not sift_vla.should_apply(sift_vla_config, layer_idx):
                    return inputs_embeds, attention_mask, position_ids

                result = sift_vla.sift_vla_prune_prefix(
                    inputs_embeds[0],
                    image_token_mask=sift_vla_config["prefix_image_token_mask"],
                    vision_token_mask=sift_vla_config["prefix_vision_token_mask"],
                    text_token_mask=sift_vla_config["prefix_text_token_mask"],
                    token_budget=int(sift_vla_config.get("token_budget", 128)),
                    prompt_token_ids=sift_vla_config.get("prompt_token_ids"),
                    text_start=sift_vla_config.get("text_start"),
                    selector_backend=str(sift_vla_config.get("selector_backend", "auto")),
                )
                if not result.applied:
                    return inputs_embeds, attention_mask, position_ids

                old_prefix_len = inputs_embeds[0].shape[1]
                suffix_len = inputs_embeds[1].shape[1]
                suffix_indices = torch.arange(
                    old_prefix_len,
                    old_prefix_len + suffix_len,
                    device=result.prefix_keep_indices.device,
                    dtype=torch.long,
                )
                combined_keep_indices = torch.cat([result.prefix_keep_indices, suffix_indices], dim=0)

                inputs_embeds = [result.prefix_hidden_states, inputs_embeds[1]]
                if attention_mask is not None:
                    attention_mask = attention_mask.index_select(-2, combined_keep_indices).index_select(
                        -1, combined_keep_indices
                    )
                if position_ids is not None:
                    position_ids = position_ids.index_select(1, combined_keep_indices)

                return inputs_embeds, attention_mask, position_ids

            # Process all layers with gradient checkpointing if enabled.
            for layer_idx in range(num_layers):
                inputs_embeds, attention_mask, position_ids = apply_sift_vla_if_needed(
                    layer_idx, inputs_embeds, attention_mask, position_ids
                )
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
