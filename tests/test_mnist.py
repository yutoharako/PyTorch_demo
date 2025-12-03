import sys
import torch
import pytest

# ensure src is importable when tests run from repo root
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pytorch_demo.data.mnist import (
    LitAutoEncoder,
    encoder as base_encoder,
    decoder as base_decoder,
)
from torch import optim, Tensor


def test_configure_optimizers_returns_adam():
    model = LitAutoEncoder(base_encoder, base_decoder)
    opt = model.configure_optimizers()
    assert isinstance(opt, optim.Adam)


def test_training_step_returns_loss_tensor():
    model = LitAutoEncoder(base_encoder, base_decoder)
    # create a small fake batch: shape (batch, channels, H, W)
    x = torch.rand(2, 1, 28, 28)
    labels = torch.zeros(2, dtype=torch.long)
    batch = (x, labels)
    loss = model.training_step(batch, 0)
    assert isinstance(loss, Tensor)
    assert loss.item() >= 0.0


def test_encoder_outputs_expected_shape():
    enc = LitAutoEncoder(base_encoder, base_decoder).encoder
    enc.eval()
    inp = torch.rand(4, 28 * 28)
    out = enc(inp)
    assert out.shape == (4, 3)
