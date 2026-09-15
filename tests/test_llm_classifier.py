from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ser.llm_classifier import FrozenLLMEmotionClassifier, resolve_emotion_token_ids


class FakeTokenizer:
    tokens = {" angry": 10, " happy": 11, " neutral": 12, " sad": 13}

    def encode(self, text, add_special_tokens=False):
        return [self.tokens[text]] if text in self.tokens else [30, 31]

    def __call__(self, text, add_special_tokens=True, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}


class FakeAudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=6)
        self.projection = nn.Linear(1, 6)
        self.encoder = SimpleNamespace(layers=nn.ModuleList([nn.Linear(6, 6) for _ in range(2)]))

    def forward(self, input_values, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.projection(input_values.unsqueeze(-1)))

    def _get_feature_vector_attention_mask(self, length, attention_mask):
        return attention_mask[:, :length]


class FakeLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.embedding = nn.Embedding(40, 8)
        self.output = nn.Linear(8, 40, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask=None, use_cache=False):
        return SimpleNamespace(logits=self.output(inputs_embeds))


def build_fake_model():
    return FrozenLLMEmotionClassifier(
        "fake-audio", "fake-llm", freeze_feature_encoder=False,
        freeze_transformer_layers=0, llm_dtype="float32",
        audio_encoder=FakeAudioEncoder(), llm=FakeLLM(), tokenizer=FakeTokenizer(),
    )


def test_verbalizer_resolution():
    assert resolve_emotion_token_ids(FakeTokenizer()) == {
        "angry": 10, "happy": 11, "neutral": 12, "sad": 13
    }


def test_verbalizer_rejects_duplicate_overrides():
    with pytest.raises(ValueError, match="distinct"):
        resolve_emotion_token_ids(FakeTokenizer(), overrides={
            "angry": 7, "happy": 7, "neutral": 8, "sad": 9
        })


def test_llm_forward_and_gradient_freezing():
    model = build_fake_model()
    model.train()
    waveform = torch.randn(2, 7)
    mask = torch.ones(2, 7, dtype=torch.long)
    logits = model(waveform, mask)
    assert logits.shape == (2, 4)
    logits.sum().backward()
    assert all(not parameter.requires_grad for parameter in model.llm.parameters())
    assert all(parameter.grad is None for parameter in model.llm.parameters())
    assert model.projector[-1].weight.grad is not None
    assert model.llm.training is False


def test_compact_state_excludes_frozen_llm_weights():
    model = build_fake_model()
    state = model.trainable_state_dict()
    assert not any(name.startswith("llm.") for name in state)
    assert "emotion_token_ids" in state
    assert any(name.startswith("projector.") for name in state)

