# Evo Neuro Bench

This repository contains a modularized implementation of the evolutionary neuro-benchmark that was originally developed inside a single Google Colab notebook. The package layout allows running experiments from notebooks while keeping the source code under version control.

## Benchmark task configuration

The high level runner (`src/evo_neuro_bench/benchmark.py`) maintains a single configuration table, `TASK_SETTINGS`, that stores the adapter requirements for each supported task. The table ensures that the number of available actions stays consistent while making it easy to toggle how logits are produced for different task families.

| Task | `n_actions` | `prefer_base_logits` | `logit_key` | Notes |
| --- | --- | --- | --- | --- |
| `train_hd_jellyfish` | 3 | `False` | – | Motor control task, uses adapter head. |
| `train_reversal` | 2 | `False` | – | Motor control task, uses adapter head. |
| `train_detour` | 3 | `False` | – | Motor control task, uses adapter head. |
| `train_local_reflex` | 3 | `False` | – | Motor control task, uses adapter head. |
| `train_rpm_mini` | 3 | `True` | `abstract_logits` | Abstract reasoning task, reuses base logits. |
| `train_arc_mini` | 3 | `True` | `abstract_logits` | Abstract reasoning task, reuses base logits. |
| `train_grid_firststep` | 3 | `True` | `abstract_logits` | Abstract reasoning task, reuses base logits. |

When adding a new task, append a corresponding entry to `TASK_SETTINGS` with the appropriate adapter parameters. For abstract tasks, prefer setting `prefer_base_logits=True` with a dedicated `logit_key`, so the benchmark can bypass the motor head while preserving the existing motor outputs for locomotion-oriented tasks.
