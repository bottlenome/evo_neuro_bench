# Development Guide

This repository packages the Evo-Neuro benchmark as a standard Python project so that the source code can be version-controlled while running experiments from Google Colab or local notebooks.

## Repository layout

```
src/evo_neuro_bench/
  utils/        # shared constants and helper modules
  models/       # biologically inspired model implementations
  adapters/     # model wrappers that expose task-specific logits
  tasks/        # datasets and training loops for each benchmark task
  benchmark.py  # orchestration helpers
```

## Recommended workflow

1. Clone the repository (or mount it in Colab) and install in editable mode:
   ```bash
   pip install -e .
   ```
2. Modify model/task modules under `src/evo_neuro_bench/`. Comments from the original notebook have been preserved for context.
3. Run smoke tests locally:
   ```bash
   python -m pytest tests
   ```
4. Commit and push your changes.

## Partial model updates

Each model lives in its own module under `src/evo_neuro_bench/models/`. Updating just one file allows you to fine-tune or retrain a single component without touching the rest of the benchmark. Re-run the corresponding task or the full benchmark using:

```python
from evo_neuro_bench import run_benchmark
run_benchmark(device="cuda", jelly_epochs=1)
```

For Colab usage, refer to `notebook/022_module_runner.ipynb`, which demonstrates cloning the repository, installing it in editable mode, and running a subset of tasks.
