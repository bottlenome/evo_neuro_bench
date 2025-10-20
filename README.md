# Evo Neuro Bench

This repository contains a modularized implementation of the evolutionary neuro-benchmark that was originally developed inside a single Google Colab notebook. The package layout allows running experiments from notebooks while keeping the source code under version control.

## Model outputs

Every brain model in `src/evo_neuro_bench/models/` now returns two parallel policy heads:

- `motor`: continuous control drive that feeds the original motor-control tasks.
- `abstract_logits`: an `N_ABSTRACT_ACTIONS (default: 8)` sized categorical head that can be reused across reasoning/ARC-like tasks.

The logits head is purposefully lightweight and derived from each model's central integrator (e.g. PFC working memory in `HumanCortexV4`). This makes it cheap to fine-tune for abstract tasks without perturbing the motor pathway.

## Task categories

Tasks are grouped into two families:

- **Motor / embodiment** (e.g. `local_reflex`, `peristalsis`, `rpm_mini`): continue to train on `motor` outputs via the auto-attached linear head inside `ModelAdapter`.
- **Abstract reasoning** (e.g. `arc_mini`, experimental `grid_firststep` variants): should consume the shared `abstract_logits` head to compare options, typically with `n_actions = N_ABSTRACT_ACTIONS`.

To switch an experiment to the abstract head, construct the adapter with `prefer_base_logits=True`:

```python
from evo_neuro_bench.adapters import ModelAdapter
from evo_neuro_bench.utils import N_ABSTRACT_ACTIONS

adapter = ModelAdapter(
    base_model,
    n_actions=N_ABSTRACT_ACTIONS,
    prefer_base_logits=True,
)
```

If a model does not provide the requested key or the dimensionality does not match `n_actions`, the adapter automatically falls back to the legacy motor head path, so existing benchmarks remain backwards compatible.
