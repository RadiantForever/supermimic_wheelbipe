# SPDX-License-Identifier: MIT
"""纯 torch 的可微前向运动学。"""

from __future__ import annotations

import torch

from .rotations import axis_angle_to_mat, quat_to_mat


class Kinematics:
    def __init__(self, model: dict, joint_names: list[str], device: str, tracked: list[str] | None = None,
                 wheel_bodies: list[str] | None = None):
        self.device = device
        self.wheel_names = list(wheel_bodies or [])
        self.joint_names = list(joint_names)
        name_to_j = {n: i for i, n in enumerate(self.joint_names)}
        bodies = model["bodies"]
        self.body_names = [b["name"] for b in bodies]
        self.parent = [int(b["parent"]) for b in bodies]

        axes, p_J, R_J, p_B, R_B, jidx, kinds = [], [], [], [], [], [], []
        self.missing_joints = []
        for b in bodies:
            kind = b["kind"]
            kinds.append(kind)
            jn = b["joint"]
            if jn is not None and jn in name_to_j:
                jidx.append(name_to_j[jn])
            else:
                if jn is not None:
                    self.missing_joints.append(jn)
                jidx.append(-1)
            axes.append(b["axis"])
            p_J.append(b["p_J"])
            R_J.append(b["q_J"])
            p_B.append(b["p_B"])
            R_B.append(b["q_B"])

        f = lambda x: torch.tensor(x, dtype=torch.float32, device=device)  # noqa: E731
        self.kind = kinds
        self.axes = f(axes)
        self.p_J = f(p_J)
        self.R_J = quat_to_mat(f(R_J))
        self.p_B = f(p_B)
        self.R_B = quat_to_mat(f(R_B))
        self.joint_idx = jidx
        self.num_bodies = len(bodies)
        self.num_movable = sum(1 for k in kinds if k in ("revolute", "prismatic"))
        if tracked is None:
            tracked = self.body_names
        self.tracked_names = list(tracked)
        # 仿真里有、但运动学树里没有的刚体：退化成根（不要因此直接崩掉整条链路）
        missing = [n for n in self.tracked_names if n not in self.body_names]
        if missing:
            print(f"[kin] 警告：这些刚体不在运动学树里，按根处理: {missing}")
        self.tracked_idx = [self.body_names.index(n) if n in self.body_names else 0 for n in self.tracked_names]

    # FK
    def forward(self, p, R, v, w, q_full, qd_full):
        """返回被跟踪刚体的世界位置/姿态/线速度/角速度，形状 (B, K, ...)。"""
        P = [None] * self.num_bodies
        Rs = [None] * self.num_bodies
        V = [None] * self.num_bodies
        W = [None] * self.num_bodies
        for i in range(self.num_bodies):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint_idx[i]
            kind = self.kind[i]
            Rp = Rs[par]
            # 关节坐标系（在父刚体中的位姿）
            pj = P[par] + (Rp @ self.p_J[i].unsqueeze(-1)).squeeze(-1)
            Rj = Rp @ self.R_J[i]
            axis_w = (Rj @ self.axes[i].unsqueeze(-1)).squeeze(-1)
            if j < 0:
                qi, qdi = None, None
            else:
                qi, qdi = q_full[..., j], qd_full[..., j]
            if kind == "revolute" and qi is not None:
                Rj2 = Rj @ axis_angle_to_mat(self.axes[i], qi)
                pj2 = pj
                wi = W[par] + axis_w * qdi.unsqueeze(-1)
                extra_v = None
            elif kind == "prismatic" and qi is not None:
                Rj2 = Rj
                pj2 = pj + axis_w * qi.unsqueeze(-1)
                wi = W[par]
                extra_v = axis_w * qdi.unsqueeze(-1)
            else:  # fixed / 缺失关节
                Rj2, pj2, wi = Rj, pj, W[par]
                extra_v = None
            # 子刚体在关节系中的偏置
            P[i] = pj2 + (Rj2 @ self.p_B[i].unsqueeze(-1)).squeeze(-1)
            Rs[i] = Rj2 @ self.R_B[i]
            vi = V[par] + torch.cross(W[par], P[i] - P[par], dim=-1)
            if extra_v is not None:
                vi = vi + extra_v
            V[i], W[i] = vi, wi
        idx = self.tracked_idx
        return (
            torch.stack([P[i] for i in idx], -2),
            torch.stack([Rs[i] for i in idx], -3),
            torch.stack([V[i] for i in idx], -2),
            torch.stack([W[i] for i in idx], -2),
        )

    # helpers
    def rest_height(self, q_full: torch.Tensor, wheel_radius: float, wheel_bodies: list[str]) -> float:
        """给定关节角，返回"轮子刚好触地"时的根刚体高度。"""
        q = q_full.reshape(1, -1).to(self.device)
        B = 1
        zero3 = torch.zeros(B, 3, device=self.device)
        eye3 = torch.eye(3, device=self.device).expand(B, 3, 3)
        P = self.forward(zero3, eye3, zero3, zero3, q, torch.zeros_like(q))[0][0]  # (K,3)
        ids = [self.tracked_names.index(n) for n in wheel_bodies if n in self.tracked_names]
        if not ids:
            return float(wheel_radius) + 0.30
        return float((-P[ids, 2]).max()) + float(wheel_radius)


@torch.no_grad()
def _fk_wheels(kin, q, ids, dev):
    B = 1
    z = torch.zeros(B, 3, device=dev)
    P = kin.forward(z, torch.eye(3, device=dev).unsqueeze(0).expand(B, 3, 3), z, z,
                    q.reshape(1, -1), torch.zeros(B, q.numel(), device=dev))[0][0]  # fmt: skip
    return P[ids]


def solve_stance(kin, spec, q0: torch.Tensor, target_h: float = 0.25, wheel_radius: float | None = None,
                 iters: int = 600, lr: float = 0.06, verbose: bool = True):
    """用可微 FK 反解一个"站立姿态"（替代全零关节角）。"""
    if wheel_radius is None:
        from .constants import WHEEL_RADIUS as wheel_radius

    dev = q0.device
    ids = [kin.tracked_names.index(n) for n in kin.wheel_names if n in kin.tracked_names]
    if len(ids) == 0:
        return q0.detach().clone(), None
    var = torch.tensor(list(spec.act_leg_idx) + list(spec.spring_idx), device=dev, dtype=torch.long)
    base = q0.detach().clone()
    qv = base[var].clone().requires_grad_(True)
    lo = spec.q_lo[var] if getattr(spec, "q_lo", None) is not None else None
    hi = spec.q_hi[var] if getattr(spec, "q_hi", None) is not None else None
    opt = torch.optim.Adam([qv], lr=lr)
    target = -(target_h - wheel_radius)  # 轮心相对 base 的 z（base 在 target_h 时轮心在 r 处）
    for _ in range(iters):
        q_full = base.clone().index_copy(0, var, qv)
        P = _fk_wheels(kin, q_full, ids, dev)
        loss = ((P[:, 2] - target) ** 2).sum() + 0.6 * (P[:, 0] ** 2).sum() \
            + 1e-3 * ((qv - base[var]) ** 2).sum()  # fmt: skip
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            if lo is not None:
                qv.clamp_(lo, hi)
    q_sol = base.clone().index_copy(0, var, qv.detach())
    q_sol = torch.maximum(torch.minimum(q_sol, spec.q_hi), spec.q_lo) if getattr(spec, "q_lo", None) is not None else q_sol
    with torch.no_grad():
        P = _fk_wheels(kin, q_sol, ids, dev)
        h = float(-P[:, 2].min()) + float(wheel_radius)
        err = float((P[:, 2] - target).abs().max())
    if verbose:
        print(f"[stance] 反解站立姿态: 目标轮心 z={target:+.3f}, 残差 {err:.3f} m, "
              f"对应根高度 {h:.3f} m, 轮心 x=[{float(P[0,0]):+.3f},{float(P[-1,0]):+.3f}]", flush=True)
    return q_sol.detach(), h
