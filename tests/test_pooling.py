import torch

from ser.models import AttentivePooling


def test_attentive_pooling_shape_and_mask() -> None:
    pooling = AttentivePooling(8)
    states = torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    output = pooling(states, mask)
    assert output.shape == (2, 8)
    assert torch.isfinite(output).all()

