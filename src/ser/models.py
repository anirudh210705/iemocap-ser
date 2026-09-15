"""CNN and pretrained speech-encoder SER models."""

from __future__ import annotations

import torch
from torch import nn


class AttentivePooling(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.score(states).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        return torch.sum(states * weights.unsqueeze(-1), dim=1)


class CNNBaseline(nn.Module):
    def __init__(self, sample_rate: int = 16_000, num_classes: int = 4, specaugment: bool = False) -> None:
        super().__init__()
        import torchaudio

        self.features = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=400, win_length=400, hop_length=160, n_mels=64
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.specaugment = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=8),
            torchaudio.transforms.TimeMasking(time_mask_param=30),
        ) if specaugment else nn.Identity()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes),
        )

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        spectrogram = self.amplitude_to_db(self.features(input_values).clamp_min(1e-10))
        if self.training:
            spectrogram = self.specaugment(spectrogram)
        mean = spectrogram.mean(dim=(-2, -1), keepdim=True)
        std = spectrogram.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        return self.network(((spectrogram - mean) / std).unsqueeze(1))


class SpeechEncoderClassifier(nn.Module):
    def __init__(
        self,
        pretrained_name: str,
        num_classes: int = 4,
        freeze_feature_encoder: bool = True,
        freeze_transformer_layers: int = 8,
        specaugment: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(pretrained_name)
        if specaugment:
            self.encoder.config.apply_spec_augment = True
            self.encoder.config.mask_time_prob = 0.05
            self.encoder.config.mask_time_length = 10
            self.encoder.config.mask_feature_prob = 0.05
            self.encoder.config.mask_feature_length = 8
        hidden_size = self.encoder.config.hidden_size
        self.pooling = AttentivePooling(hidden_size)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden_size, num_classes))
        if freeze_feature_encoder and hasattr(self.encoder, "freeze_feature_encoder"):
            self.encoder.freeze_feature_encoder()
        layers = self._transformer_layers()
        for layer in layers[:freeze_transformer_layers]:
            for parameter in layer.parameters():
                parameter.requires_grad = False

    def _transformer_layers(self) -> list[nn.Module]:
        candidates = (
            getattr(getattr(self.encoder, "encoder", None), "layers", None),
            getattr(getattr(self.encoder, "encoder", None), "layer", None),
        )
        for candidate in candidates:
            if candidate is not None:
                return list(candidate)
        return []

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        output = self.encoder(input_values=input_values, attention_mask=attention_mask)
        feature_mask = None
        if attention_mask is not None and hasattr(self.encoder, "_get_feature_vector_attention_mask"):
            feature_mask = self.encoder._get_feature_vector_attention_mask(output.last_hidden_state.shape[1], attention_mask)
        pooled = self.pooling(output.last_hidden_state, feature_mask)
        return self.classifier(pooled)


def build_model(config: dict[str, object]) -> nn.Module:
    if config["model"] == "cnn":
        return CNNBaseline(
            sample_rate=int(config.get("sample_rate", 16_000)),
            specaugment=config.get("augmentation") == "specaugment",
        )
    if config["model"] in {"wav2vec2", "hubert"}:
        return SpeechEncoderClassifier(
            pretrained_name=str(config["pretrained_name"]),
            freeze_feature_encoder=bool(config.get("freeze_feature_encoder", True)),
            freeze_transformer_layers=int(config.get("freeze_transformer_layers", 0)),
            specaugment=config.get("augmentation") == "specaugment",
        )
    if config["model"] == "wav2vec2_llm":
        from .llm_classifier import DEFAULT_PROMPT, FrozenLLMEmotionClassifier

        return FrozenLLMEmotionClassifier(
            audio_encoder_name=str(config["pretrained_name"]),
            llm_name=str(config["llm_name"]),
            prompt=str(config.get("prompt", DEFAULT_PROMPT)),
            verbalizers=config.get("emotion_verbalizers"),
            emotion_token_overrides=config.get("emotion_token_overrides"),
            freeze_feature_encoder=bool(config.get("freeze_feature_encoder", True)),
            freeze_transformer_layers=int(config.get("freeze_transformer_layers", 0)),
            projector_dropout=float(config.get("projector_dropout", 0.1)),
            llm_dtype=str(config.get("llm_dtype", "float16")),
            specaugment=config.get("augmentation") == "specaugment",
            gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
        )
    raise ValueError(f"Unsupported model: {config['model']}")
