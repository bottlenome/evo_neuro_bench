"""High level benchmark runner for Evo-Neuro models."""

from typing import Dict

import torch

from .adapters import ModelAdapter
from .models import build_models
from .tasks import (
    train_arc_mini,
    train_detour,
    train_grid_firststep,
    train_hd_jellyfish,
    train_local_reflex,
    train_reversal,
    train_rpm_mini,
)
from .utils import compute_efficiency, compute_tal_metrics, set_seed


def run_benchmark(device: str | None = None, jelly_epochs=3, rev_steps=1500, batch_size=128, detour_epochs=15):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    set_seed(0)
    results: Dict[str, Dict[str, float | None]] = {}
    models = build_models(device=device)
    for name, base in models.items():
        params = sum(p.numel() for p in base.parameters())
        init_state = {k: v.detach().clone() for k, v in base.state_dict().items()}

        print("\n" + "=" * 60)
        print(f"Model: {name}")
        # Jellyfish
        base.load_state_dict(init_state, strict=True)
        agent = ModelAdapter(base, n_actions=3, use_memory=False).to(device)
        jelly_acc, jelly_curve = train_hd_jellyfish(agent, device=device, epochs=jelly_epochs, batch_size=batch_size)
        j_ttc, j_auc, j_asy = compute_tal_metrics(jelly_curve, criterion=0.90, window=100, budget=2000)
        j_eff = compute_efficiency(j_auc, params, trials=2000)

        # Reversal
        base.load_state_dict(init_state, strict=True)
        agent = ModelAdapter(base, n_actions=2, use_memory=False).to(device)
        pre, post, rev_curve = train_reversal(agent, device=device, steps=rev_steps)
        rev_ttc, rev_auc, rev_asy = compute_tal_metrics(rev_curve, criterion=0.80, window=200, budget=rev_steps)
        rev_eff = compute_efficiency(rev_auc, params, trials=rev_steps)

        # Detour
        base.load_state_dict(init_state, strict=True)
        agent = ModelAdapter(base, n_actions=3, use_memory=False).to(device)
        detour_acc, detour_curve = train_detour(agent, device=device, epochs=detour_epochs, batch_size=batch_size)
        detour_ttc, detour_auc, detour_asy = compute_tal_metrics(detour_curve, criterion=0.85, window=200, budget=3000)
        detour_eff = compute_efficiency(detour_auc, params, trials=3000)

        # Local reflex
        base.load_state_dict(init_state, strict=True)
        agent = ModelAdapter(base, n_actions=3, use_memory=False).to(device)
        reflex_acc, reflex_curve = train_local_reflex(agent, device=device, epochs=15)
        reflex_ttc, reflex_auc, reflex_asy = compute_tal_metrics(reflex_curve, criterion=0.85, window=200, budget=3000)
        reflex_eff = compute_efficiency(reflex_auc, params, trials=3000)

        results[name] = {
            "jelly_acc": jelly_acc,
            "jelly_TTC": j_ttc,
            "jelly_AUC": j_auc,
            "jelly_Asy": j_asy,
            "jelly_Eff": j_eff,
            "rev_pre": pre,
            "rev_post": post,
            "rev_TTC": rev_ttc,
            "rev_AUC": rev_auc,
            "rev_Asy": rev_asy,
            "rev_Eff": rev_eff,
            "detour_acc": detour_acc,
            "detour_TTC": detour_ttc,
            "detour_AUC": detour_auc,
            "detour_Asy": detour_asy,
            "detour_Eff": detour_eff,
            "reflex_acc": reflex_acc,
            "reflex_TTC": reflex_ttc,
            "reflex_AUC": reflex_auc,
            "reflex_Asy": reflex_asy,
            "reflex_Eff": reflex_eff,
        }

        # RPM-Mini
        base.load_state_dict(init_state, strict=True)
        agent = ModelAdapter(base, n_actions=3, use_memory=False).to(device)
        rpm_acc, rpm_curve = train_rpm_mini(agent, device=device, epochs=80, batch_size=batch_size)
        rpm_ttc, rpm_auc, rpm_asy = compute_tal_metrics(rpm_curve, criterion=0.75, window=200, budget=3000)

        # ARC-Mini
        base.load_state_dict(init_state, strict=True)
        agent = ModelAdapter(base, n_actions=3, use_memory=False).to(device)
        arc_acc, arc_curve = train_arc_mini(agent, device=device, epochs=80, batch_size=batch_size)
        arc_ttc, arc_auc, arc_asy = compute_tal_metrics(arc_curve, criterion=0.75, window=200, budget=3000)

        # Grid-FirstStep
        base.load_state_dict(init_state, strict=True)
        agent = ModelAdapter(base, n_actions=3, use_memory=False).to(device)
        gpf_acc, gpf_curve = train_grid_firststep(agent, device=device, epochs=50, batch_size=batch_size)
        gpf_ttc, gpf_auc, gpf_asy = compute_tal_metrics(gpf_curve, criterion=0.75, window=200, budget=3000)

        results[name].update(
            {
                "rpm_acc": rpm_acc,
                "rpm_TTC": rpm_ttc,
                "rpm_AUC": rpm_auc,
                "rpm_Asy": rpm_asy,
                "arc_acc": arc_acc,
                "arc_TTC": arc_ttc,
                "arc_AUC": arc_auc,
                "arc_Asy": arc_asy,
                "gpf_acc": gpf_acc,
                "gpf_TTC": gpf_ttc,
                "gpf_AUC": gpf_auc,
                "gpf_Asy": gpf_asy,
            }
        )

    # Summary
    print("\n" + "#" * 60)
    print("# Summary: jelly_acc | rev_pre | rev_post| detour_acc")
    print("#" * 60)
    for k, r in results.items():
        print(k)
        print({
            "jelly_acc": r["jelly_acc"],
            "rev_pre": r["rev_pre"],
            "rev_post": r["rev_post"],
            "detour_acc": r["detour_acc"],
        })
    return results


__all__ = ["run_benchmark"]
