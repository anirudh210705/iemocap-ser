"""Frozen causal-LLM classification head driven by one projected acoustic token."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from . import LABEL_NAMES
from .models import AttentivePooling


DEFAULT_PROMPT = (
    "Classify the emotion expressed in the preceding speech. "
    "Possible emotions: angry, happy, neutral, sad. Answer with one emotion:"
)


def resolve_emotion_token_ids(
    tokenizer: Any,
    verbalizers: Mapping[str, str] | None = None,
    overrides: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Resolve one distinct vocabulary token for each class.

    A leading-space spelling is preferred because decoder tokenizers commonly
    encode word continuations differently from sentence-initial words.
    """
    verbalizers = verbalizers or {label: label for label in LABEL_NAMES}
    overrides = overrides or {}
    missing = set(LABEL_NAMES) - set(verbalizers)
    if missing:
        raise ValueError(f"Missing emotion verbalizers: {sorted(missing)}")

    result: dict[str, int] = {}
    failures: list[str] = []
    for label in LABEL_NAMES:
        if label in overrides:
            result[label] = int(overrides[label])
            continue
        word = str(verbalizers[label])
        candidates = (f" {word}", word)
        token_id = None
        diagnostics = []
        for candidate in candidates:
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            diagnostics.append(f"{candidate!r}->{ids}")
            if len(ids) == 1:
                token_id = int(ids[0])
                break
        if token_id is None:
            failures.append(f"{label}: " + ", ".join(diagnostics))
        else:
            result[label] = token_id
    if failures:
        raise ValueError(
            "Each emotion verbalizer must map to one token. Set emotion_token_overrides "
            "or choose different words. " + "; ".join(failures)
        )
    if len(set(result.values())) != len(result):
        raise ValueError(f"Emotion verbalizers must use distinct token IDs: {result}")
    return result


def torch_dtype(name: str) -> torch.dtype:
    options = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in options:
        raise ValueError(f"llm_dtype must be one of {sorted(options)}")
    return options[name]


class FrozenLLMEmotionClassifier(nn.Module):
    """Use a frozen decoder-only LLM as a four-way emotion classifier.

    The LLM is frozen but deliberately *not* run under ``no_grad``: gradients
    must pass through it to the projected acoustic token.
    """

    def __init__(
        self,
        audio_encoder_name: str,
        llm_name: str,
        prompt: str = DEFAULT_PROMPT,
        verbalizers: Mapping[str, str] | None = None,
        emotion_token_overrides: Mapping[str, int] | None = None,
        freeze_feature_encoder: bool = True,
        freeze_transformer_layers: int = 8,
        projector_dropout: float = 0.1,
        llm_dtype: str = "float16",
        specaugment: bool = False,
        gradient_checkpointing: bool = False,
        *,
        audio_encoder: nn.Module | None = None,
        llm: nn.Module | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        super().__init__()
        if audio_encoder is None or llm is None or tokenizer is None:
            from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

            audio_encoder = audio_encoder or AutoModel.from_pretrained(audio_encoder_name)
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(llm_name)
            llm = llm or AutoModelForCausalLM.from_pretrained(
                llm_name, torch_dtype=torch_dtype(llm_dtype), low_cpu_mem_usage=True
            )
        self.audio_encoder = audio_encoder
        self.llm = llm
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.llm_name = llm_name

        if specaugment:
            self.audio_encoder.config.apply_spec_augment = True
            self.audio_encoder.config.mask_time_prob = 0.05
            self.audio_encoder.config.mask_time_length = 10
        if gradient_checkpointing and hasattr(self.audio_encoder, "gradient_checkpointing_enable"):
            self.audio_encoder.gradient_checkpointing_enable()
        if freeze_feature_encoder and hasattr(self.audio_encoder, "freeze_feature_encoder"):
            self.audio_encoder.freeze_feature_encoder()
        layers = self._audio_transformer_layers()
        for layer in layers[:freeze_transformer_layers]:
            for parameter in layer.parameters():
                parameter.requires_grad = False

        for parameter in self.llm.parameters():
            parameter.requires_grad = False
        self.llm.eval()
        if hasattr(self.llm.config, "use_cache"):
            self.llm.config.use_cache = False

        audio_size = int(self.audio_encoder.config.hidden_size)
        embedding_layer = self.llm.get_input_embeddings()
        llm_size = int(embedding_layer.embedding_dim)
        self.pooling = AttentivePooling(audio_size)
        self.projector = nn.Sequential(
            nn.LayerNorm(audio_size),
            nn.Dropout(projector_dropout),
            nn.Linear(audio_size, llm_size),
        )

        token_map = resolve_emotion_token_ids(tokenizer, verbalizers, emotion_token_overrides)
        self.emotion_token_map = token_map
        self.register_buffer(
            "emotion_token_ids",
            torch.tensor([token_map[label] for label in LABEL_NAMES], dtype=torch.long),
            persistent=True,
        )
        prompt_encoding = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
        self.register_buffer("prompt_input_ids", prompt_encoding["input_ids"], persistent=True)
        self.register_buffer(
            "prompt_attention_mask",
            prompt_encoding.get("attention_mask", torch.ones_like(prompt_encoding["input_ids"])),
            persistent=True,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # A frozen LLM should never enable dropout during task training.
        self.llm.eval()
        return self

    def _audio_transformer_layers(self) -> list[nn.Module]:
        candidates = (
            getattr(getattr(self.audio_encoder, "encoder", None), "layers", None),
            getattr(getattr(self.audio_encoder, "encoder", None), "layer", None),
        )
        for candidate in candidates:
            if candidate is not None:
                return list(candidate)
        return []

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        audio_output = self.audio_encoder(input_values=input_values, attention_mask=attention_mask)
        feature_mask = None
        if attention_mask is not None and hasattr(self.audio_encoder, "_get_feature_vector_attention_mask"):
            feature_mask = self.audio_encoder._get_feature_vector_attention_mask(
                audio_output.last_hidden_state.shape[1], attention_mask
            )
        pooled = self.pooling(audio_output.last_hidden_state, feature_mask)

        embedding_layer = self.llm.get_input_embeddings()
        prompt_ids = self.prompt_input_ids.expand(input_values.shape[0], -1)
        prompt_mask = self.prompt_attention_mask.expand(input_values.shape[0], -1)
        prompt_embeddings = embedding_layer(prompt_ids).detach()
        acoustic_token = self.projector(pooled).to(prompt_embeddings.dtype).unsqueeze(1)
        inputs_embeds = torch.cat((acoustic_token, prompt_embeddings), dim=1)
        acoustic_mask = torch.ones(
            (input_values.shape[0], 1), dtype=prompt_mask.dtype, device=prompt_mask.device
        )
        llm_mask = torch.cat((acoustic_mask, prompt_mask), dim=1)
        output = self.llm(inputs_embeds=inputs_embeds, attention_mask=llm_mask, use_cache=False)
        vocabulary_logits = output.logits[:, -1, :]
        return vocabulary_logits.index_select(-1, self.emotion_token_ids)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        # Persistent non-parameter buffers capture prompt and label-token IDs.
        buffer_names = {"emotion_token_ids", "prompt_input_ids", "prompt_attention_mask"}
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable or name in buffer_names
        }

    def parameter_summary(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
