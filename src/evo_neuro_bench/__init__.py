"""Evo-Neuro benchmark package."""

from __future__ import annotations

from typing import Any

from .utils import set_seed

__all__ = ["run_benchmark", "build_models", "set_seed"]


def run_benchmark(*args: Any, **kwargs: Any) -> Any:
    """Proxy for :func:`evo_neuro_bench.benchmark.run_benchmark`.

    The benchmark module pulls in a number of heavy dependencies (for
    example :mod:`torch`).  When the package is imported in lightweight
    environments—such as Google Colab before ``torch`` is installed—we still
    want attribute access like ``evo_neuro_bench.run_benchmark`` and
    ``evo_neuro_bench.build_models`` to succeed.  Importing the benchmark
    module lazily avoids failing the package import while keeping the public
    API unchanged.
    """

    from .benchmark import run_benchmark as _run_benchmark

    return _run_benchmark(*args, **kwargs)


def build_models(*args: Any, **kwargs: Any) -> Any:
    """Proxy for :func:`evo_neuro_bench.models.build_models`.

    Lazily importing the models registry means the top-level package import
    no longer fails when optional runtime dependencies are missing.  This
    matches user expectations on Colab where they may import the package
    before installing the heavy dependencies.
    """

    from .models import build_models as _build_models

    return _build_models(*args, **kwargs)
