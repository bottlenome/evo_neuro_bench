"""Minimal smoke tests for evo_neuro_bench package."""

import pytest


def test_models_forward_cpu():
    torch = pytest.importorskip("torch")
    from evo_neuro_bench.models import build_models
    from evo_neuro_bench.utils import make_obs

    device = "cpu"
    models = build_models(device=device)
    obs = make_obs(batch=2, device=device)
    for model in models.values():
        out = model(obs)
        assert "motor" in out
        assert out["motor"].shape[0] == 2
        assert torch.isfinite(out["motor"]).all()
