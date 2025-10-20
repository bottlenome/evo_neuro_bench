"""Minimal smoke tests for evo_neuro_bench package."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_models_forward_cpu():
    torch = pytest.importorskip("torch")
    from evo_neuro_bench.models import build_models
    from evo_neuro_bench.utils import N_ABSTRACT_ACTIONS, make_obs

    device = "cpu"
    models = build_models(device=device)
    obs = make_obs(batch=2, device=device)
    for model in models.values():
        out = model(obs)
        assert "motor" in out
        assert out["motor"].shape[0] == 2
        assert torch.isfinite(out["motor"]).all()
        assert "abstract_logits" in out
        assert out["abstract_logits"].shape == (2, N_ABSTRACT_ACTIONS)
        assert torch.isfinite(out["abstract_logits"]).all()


def test_model_adapter_motor_and_base_logits_paths():
    torch = pytest.importorskip("torch")

    class DummyBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = torch.nn.Linear(3, 5)
            self.motor_head = torch.nn.Linear(5, 4)
            self.abstract_head = torch.nn.Linear(5, 6)

        def forward(self, obs):
            x = obs["stim"]
            feat = torch.tanh(self.trunk(x))
            motor = torch.tanh(self.motor_head(feat))
            abstract_logits = self.abstract_head(feat)
            return {"motor": motor, "abstract_logits": abstract_logits, "feat": feat}

    from evo_neuro_bench.adapters import ModelAdapter

    base = DummyBase()
    adapter = ModelAdapter(base, n_actions=7)
    obs = {"stim": torch.randn(8, 3)}
    logits, raw = adapter(obs)
    assert logits.shape == (8, 7)
    assert adapter._head is not None

    loss = logits.sum()
    loss.backward()
    assert base.motor_head.weight.grad is not None
    assert base.motor_head.weight.grad.abs().sum() > 0
    # abstract head is not used when prefer_base_logits=False
    if base.abstract_head.weight.grad is not None:
        assert torch.allclose(base.abstract_head.weight.grad, torch.zeros_like(base.abstract_head.weight.grad))

    base = DummyBase()
    adapter = ModelAdapter(base, n_actions=6, prefer_base_logits=True)
    obs = {"stim": torch.randn(4, 3)}
    logits, raw = adapter(obs)
    assert logits.shape == (4, 6)
    assert torch.allclose(logits, raw["abstract_logits"])
    assert adapter._head is None

    loss = logits.pow(2).mean()
    loss.backward()
    assert base.abstract_head.weight.grad is not None
    assert base.abstract_head.weight.grad.abs().sum() > 0
