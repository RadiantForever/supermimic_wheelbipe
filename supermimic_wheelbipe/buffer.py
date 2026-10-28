# SPDX-License-Identifier: MIT
"""GPU 定长回放缓冲，布局 (time, env)（与 supermimic 一致，仅维度参数化）。"""

from __future__ import annotations

import torch


class ReplayBuffer:
    def __init__(self, T: int, N: int, device: str, state_dim: int, act_dim: int, xi_dim: int):
        self.T, self.N, self.device = T, N, device
        f = torch.float32
        self.fields = {
            "s": (state_dim, f),
            "s1": (state_dim, f),
            "a": (act_dim, f),
            "a_prev": (act_dim, f),
            "xi": (xi_dim, f),
            "r": ((), f),
            "f": ((), torch.long),
            "ep_step": ((), torch.int32),
            "done": ((), torch.bool),
        }
        self.data = {}
        for k, (shape, dt) in self.fields.items():
            shp = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[k] = torch.zeros(shp, dtype=dt, device=device)
        self.ptr, self.size = 0, 0

    def add(self, tr: dict):
        for k in self.fields:
            self.data[k][self.ptr] = tr[k]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        return (self.ptr - self.size + age) % self.T

    def sample_windows(self, B: int, k: int):
        """k 步窗口；返回 (B, k, ...) 与有效掩码 (B, k)。"""
        k = max(1, min(k, self.size - 1))
        t0 = torch.randint(0, self.size - k + 1, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        ages = t0.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        e = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, e] for key in ("s", "s1", "a", "xi", "done")}
        d = out["done"].float()
        mask = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        out["mask"] = mask
        return out

    def sample_starts(self, B: int, hist: int):
        """想象起点 + 其观测历史索引（由旧到新）。"""
        age = torch.randint(0, max(self.size, 1), (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))
        back = torch.minimum(back, age.unsqueeze(-1))
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        start = {k: self.data[k][row, env] for k in ("s", "a_prev", "xi", "f")}
        hist_d = {k: self.data[k][hrows, henv] for k in ("s", "a_prev", "xi", "f")}
        return start, hist_d
