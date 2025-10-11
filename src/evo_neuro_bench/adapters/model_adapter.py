"""Adapter that converts raw motor outputs into task-specific logits."""

from typing import Dict, Tuple

import torch
import torch.nn as nn

__all__ = ["ModelAdapter"]


# ===============================================================
# Adapter: auto head to n_actions; optional tiny memory (off by default)
# ===============================================================
class ModelAdapter(nn.Module):
    def __init__(self, base: nn.Module, n_actions: int, use_memory: bool = False, mem_dim: int = 64):
        super().__init__()
        self.base = base
        self.n_actions = n_actions
        self.use_memory = use_memory
        self.mem_dim = mem_dim
        self._head = None
        self._mem = None
        self._h = None

    def _lazy_init(self, motor_dim, device):
        if self.use_memory:
            self._mem = nn.GRUCell(motor_dim, self.mem_dim).to(device)
            self._head = nn.Linear(self.mem_dim, self.n_actions).to(device)
        else:
            self._head = nn.Linear(motor_dim, self.n_actions).to(device)

    def reset_memory(self, B: int = 1, device: str = "cpu") -> None:
        if self.use_memory and self._mem is not None:
            self._h = torch.zeros(B, self.mem_dim, device=device)

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  # type: ignore[override]
        out = self.base(obs)
        motor = out["motor"]
        if self._head is None:
            self._lazy_init(motor.size(-1), motor.device)
        if self.use_memory:
            if (self._h is None) or (self._h.size(0) != motor.size(0)):
                self.reset_memory(B=motor.size(0), device=motor.device)
            self._h = self._mem(motor, self._h)
            logits = self._head(self._h)
        else:
            logits = self._head(motor)
        return logits, out
