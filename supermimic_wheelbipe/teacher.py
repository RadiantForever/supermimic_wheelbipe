# SPDX-License-Identifier: MIT
"""老师策略（wheeled-legged_RL 里已经训练好的 PPO 策略）。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import torch
import torch.nn as nn

TEACHER_LEG_KEYS = (".*_rear1_joint", ".*_front1_joint")
TEACHER_WHEEL_KEYS = (".*_wheel_joint",)
TEACHER_OBS_DIM = 35
TEACHER_ACT_DIM = 6


def resolve_names(keys, names: list[str]) -> tuple[list[int], list[str]]:
    """等价于 Isaac Lab 的 ``resolve_matching_names``：按 **articulation 顺序** 输出。"""
    if isinstance(keys, str):
        keys = [keys]
    idx, out = [], []
    for i, n in enumerate(names):
        for k in keys:
            if re.fullmatch(k, n):
                idx.append(i)
                out.append(n)
                break
    return idx, out


@dataclass
class TeacherObsCfg:
    ang_vel_scale: float = 0.5      # 训练时 env.yaml: obs_input_scale_cfg.root_ang_vel_b = 0.5
    cmd_ang_scale: float = 0.5      # obs_input_scale_cfg.command.ang_vel_z = 0.5
    gravity_scale: float = 1.0
    joint_pos_scale: float = 1.0
    joint_vel_scale: float = 0.1
    height_scale: float = 5.0
    height_clip: tuple = (0.0, 1.0)
    clip: float = 100.0
    joint_vel_clip: float = 200.0
    ctrl_mode: tuple = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _build_mlp_from_state_dict(sd: dict, prefix: str = "actor.", act: str = "elu") -> nn.Sequential:
    """从 rsl_rl 的 state_dict 重建 actor（层索引 0/2/4/... 是 Linear）。"""
    lin = sorted({int(k.split(".")[1]) for k in sd if k.startswith(prefix) and k.endswith(".weight")})
    if not lin:
        raise KeyError(f"state_dict 里没有 {prefix}*.weight")
    act_cls = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}.get(act.lower(), nn.ELU)
    layers: list[nn.Module] = []
    for i, li in enumerate(lin):
        w = sd[f"{prefix}{li}.weight"]
        layers.append(nn.Linear(w.shape[1], w.shape[0]))
        if i < len(lin) - 1:
            layers.append(act_cls())
    net = nn.Sequential(*layers)
    net.load_state_dict({k.replace(prefix, ""): v for k, v in sd.items() if k.startswith(prefix)})
    return net


def _read_activation(run_dir: str) -> str:
    p = os.path.join(run_dir, "params", "agent.yaml")
    if os.path.exists(p):
        m = re.search(r"activation:\s*(\w+)", open(p).read())
        if m:
            return m.group(1)
    return "elu"


def find_pretrained(root: str | None = None) -> list[str]:
    """列出可用的预训练 run 目录（含 exported/policy.pt 或 model_*.pt）。"""
    if root is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        root = os.path.join(here, "..", "wheeled-legged_RL", "pretrained")
    root = os.path.normpath(root)
    if not os.path.isdir(root):
        return []
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        has_jit = os.path.exists(os.path.join(dirpath, "exported", "policy.pt"))
        has_ckpt = any(re.fullmatch(r"model_\d+\.pt", f) for f in filenames)
        if has_jit or has_ckpt:
            out.append(dirpath)
    return sorted(out)


def pick_best_teacher(arg: str | None = None) -> str:
    """``auto`` -> 优先 flat_and_rotation 且有 JIT 导出的最新 run。"""
    if arg and arg != "auto":
        return arg
    runs = find_pretrained()
    if not runs:
        raise FileNotFoundError("找不到预训练权重，请用 --teacher 显式指定 run 目录")
    for key in ("flat_and_rotation", "rotation_translation", "rough_dash", "rough_rotation_stair"):
        cands = [r for r in runs if key in r and os.path.exists(os.path.join(r, "exported", "policy.pt"))]
        if cands:
            return sorted(cands)[-1]
    return runs[-1]


def load_teacher_net(ckpt: str, device: str = "cuda:0") -> nn.Module:
    """支持三种来源：exported/policy.pt (TorchScript)、model_XXXX.pt (rsl_rl)、目录（自动挑）。"""
    if os.path.isdir(ckpt):
        jit = os.path.join(ckpt, "exported", "policy.pt")
        models = sorted(
            [f for f in os.listdir(ckpt) if re.fullmatch(r"model_\d+\.pt", f)],
            key=lambda s: int(re.findall(r"\d+", s)[0]),
        )
        if os.path.exists(jit):
            ckpt = jit
        elif models:
            ckpt = os.path.join(ckpt, models[-1])
        else:
            raise FileNotFoundError(f"{ckpt} 下既没有 exported/policy.pt 也没有 model_*.pt")
    if ckpt.endswith(".pt"):
        try:  # 先按 TorchScript 试
            net = torch.jit.load(ckpt, map_location=device)
            net.eval()
            return net
        except Exception:  # noqa: BLE001
            pass
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck)
        net = _build_mlp_from_state_dict(sd, "actor.", _read_activation(os.path.dirname(ckpt)))
        return net.to(device).eval()
    raise ValueError(f"不认识的老是权重文件: {ckpt}")


class Teacher:
    """把老师策略包装成"喂我们的状态 → 出和我们同语义的 6 维动作"。"""

    def __init__(self, spec, joint_names: list[str], default_q: torch.Tensor, ckpt: str,
                 device: str = "cuda:0", cfg: TeacherObsCfg | None = None, verbose: bool = True):
        self.cfg = cfg or TeacherObsCfg()
        self.device = device
        self.net = load_teacher_net(ckpt, device)
        self.path = ckpt
        names = list(joint_names)
        self.leg_idx, self.leg_names = resolve_names(TEACHER_LEG_KEYS, names)
        self.wheel_idx, self.wheel_names = resolve_names(TEACHER_WHEEL_KEYS, names)
        if len(self.leg_idx) != 4 or len(self.wheel_idx) != 2:
            raise RuntimeError(f"老师需要 4 腿 + 2 轮，实际 leg={self.leg_names} wheel={self.wheel_names}")
        # 我们的状态是 (nq, nd) 压缩过的：full articulation index -> 状态列号
        q_pos = {j: i for i, j in enumerate(spec.q_idx)}
        qd_pos = {j: i for i, j in enumerate(spec.qd_idx)}
        self.leg_qcols = torch.tensor([q_pos[j] for j in self.leg_idx], dtype=torch.long, device=device)
        self.leg_dcols = torch.tensor([qd_pos[j] for j in self.leg_idx], dtype=torch.long, device=device)
        self.wheel_dcols = torch.tensor([qd_pos[j] for j in self.wheel_idx], dtype=torch.long, device=device)
        self.default_leg = default_q[self.leg_qcols].clone()
        self.obs_dim = TEACHER_OBS_DIM
        self.na = TEACHER_ACT_DIM
        self.prev_action: torch.Tensor | None = None
        self._check_io()
        if verbose:
            print(f"[teacher] {ckpt}")
            print(f"[teacher] legs={self.leg_names} wheels={self.wheel_names} obs={self.obs_dim} act={self.na}")

    def _check_io(self):
        n = torch.zeros(1, TEACHER_OBS_DIM, device=self.device)
        try:
            out = self.net(n)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"老师网络不接受 {TEACHER_OBS_DIM} 维输入（{e}）；检查是不是选了别的机器人的权重"
            ) from e
        if out.shape[-1] != TEACHER_ACT_DIM:
            raise RuntimeError(f"老师输出 {out.shape[-1]} 维，期望 {TEACHER_ACT_DIM}")

    # obs
    def build_obs(self, S, cmd: torch.Tensor) -> torch.Tensor:
        """S = (p,R,v,w,q,qd)，cmd = (N,4) [vx, vy, wz, height]（与我们的指令同格式）。"""
        c = self.cfg
        _, R, _, w, q, qd = S
        B = R.shape[0]
        if self.prev_action is None or self.prev_action.shape[0] != B:
            self.prev_action = torch.zeros(B, TEACHER_ACT_DIM, device=self.device)
        Rt = R.transpose(-1, -2)
        ang_vel_b = (Rt @ w.unsqueeze(-1)).squeeze(-1) * c.ang_vel_scale
        grav_b = -R[..., 2, :] * c.gravity_scale
        leg_pos = (q.index_select(1, self.leg_qcols) - self.default_leg) * c.joint_pos_scale
        jpos = torch.cat((leg_pos, torch.zeros(B, 2, device=self.device)), -1)  # 轮子位置恒置 0
        jvel = torch.cat((qd.index_select(1, self.leg_dcols), qd.index_select(1, self.wheel_dcols)), -1)
        jvel = jvel.clamp(-c.joint_vel_clip, c.joint_vel_clip) * c.joint_vel_scale
        cmd_obs = torch.stack((cmd[:, 0], cmd[:, 1], cmd[:, 2] * c.cmd_ang_scale), -1).clamp(-c.clip, c.clip)
        h = cmd[:, 3:4].clamp(*c.height_clip) * c.height_scale
        ctrl = torch.tensor(c.ctrl_mode, device=self.device, dtype=cmd.dtype).unsqueeze(0).expand(B, -1)
        obs = torch.cat(
            (
                cmd_obs,
                h,
                ang_vel_b.clamp(-c.clip, c.clip),
                grav_b.clamp(-c.clip, c.clip),
                jpos.clamp(-c.clip, c.clip),
                jvel,
                self.prev_action.clamp(-c.clip, c.clip),
                ctrl,
            ),
            -1,
        )
        return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    # 动作
    @torch.no_grad()
    def act(self, S, cmd: torch.Tensor, update_prev: bool = True) -> torch.Tensor:
        obs = self.build_obs(S, cmd)
        a = self.net(obs)
        if update_prev:
            self.prev_action = a.detach().clone()
        return a.detach()

    def reset(self, env_ids: torch.Tensor | None = None):
        if self.prev_action is None:
            return
        if env_ids is None:
            self.prev_action[:] = 0.0
        else:
            self.prev_action[env_ids] = 0.0
