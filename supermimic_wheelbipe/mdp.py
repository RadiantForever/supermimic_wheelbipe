# SPDX-License-Identifier: MIT
"""解析式 MDP（观测 / 奖励 / 终止 / 评论家特征）。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .constants import ACTION_SCALE_LEG, ACTION_SCALE_WHEEL, CONTROL_DT, SPRING_BIAS_RANGE, WHEEL_VEL_LIMIT
from .rotations import yaw_mat, mat_to_6d

# 各类 actuator 的 PD 参数（用于奖励里的力矩估计，与 robot.py 一致）
KP_LEG, KD_LEG, EFF_LEG = 60.0, 2.0, 40.0
KD_WHEEL, EFF_WHEEL = 0.2, 5.0

# 单步可获得的最大正奖励（5 个 +1 的指数项），用于归一化回报
MAX_STEP_REWARD = 5.0 * CONTROL_DT
XI_DIM = 5  # [mu, spring_bias, com(3)]


@dataclass
class RewardCfg:
    # —— 跟踪（sigma 全部取自 wheeled-legged_RL 的 V14 flat 配置）——
    w_track_lin: float = 1.0
    sigma_lin: float = 0.5
    lin_err_clip: float = 1.0
    # 试过 -5.0（加强大误差梯度）：实测顶速 1.39→1.30、1.6/1.8 档误差反而变大，
    # 因为"跑快"的代价放大后最优策略变成"干脆别跑"。退回原值。
    w_track_lin_sq: float = -1.0
    sigma_lin_sq: float = 0.25
    w_track_ang: float = 1.0
    sigma_ang: float = 0.25
    ang_err_clip: float = 0.8
    w_track_ang_sq: float = -1.0
    sigma_ang_sq: float = 0.5
    w_height: float = 1.0
    sigma_height: float = 0.001
    height_err_clip: float = 0.15
    w_height_sq: float = -1.0
    sigma_height_sq: float = 10.0
    # —— 姿态 ——
    w_roll_exp: float = 1.0
    sigma_roll: float = 0.02
    w_pitch_exp: float = 1.0
    sigma_pitch: float = 0.01
    w_roll_v: float = -2.0
    w_pitch_v: float = -2.0
    orient_v_A: float = 2.0
    orient_v_bias: float = 2.0
    orient_v_sigma: float = 3.0
    # —— 正则 ——
    w_lin_vel_z: float = -0.5
    w_ang_vel_xy: float = -0.05
    w_action_rate: float = -0.01
    w_leg_vel: float = -5e-3
    w_wheel_vel: float = -1e-5
    w_wheel_power: float = -1e-4
    w_joint_torque: float = -1e-4
    w_no_fork: float = -1.0
    no_fork_distance: float = 0.05
    no_fork_sigma: float = 5.0
    # 终止惩罚 -20：-200 会让 bin 采样权重爆 40 倍
    w_termination: float = -20.0
    term_height: float = 0.15          # 根高度低于该值 ≈ base_link 触地
    term_tilt_cos: float = 0.766       # cos(40°)
    term_speed: float = 20.0


def invariant_features(S) -> torch.Tensor:
    """航向与平面位置无关的状态特征（世界模型输入用）。"""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(H.transpose(-1, -2) @ R),
            (Rt @ v.unsqueeze(-1)).squeeze(-1),
            (Rt @ w.unsqueeze(-1)).squeeze(-1),
            q,
            qd,
        ),
        -1,
    )


class MDP:
    def __init__(self, spec, cmd, kin=None, default_q=None, cfg: RewardCfg | None = None, xi_dim: int = XI_DIM,
                 qd_limit=None):
        self.spec, self.cmd, self.kin = spec, cmd, kin
        self.cfg = cfg or RewardCfg()
        self.dev = cmd.cmd.device
        self.xi_dim = xi_dim
        self.nq, self.nd, self.na, self.nj = spec.nq, spec.nd, spec.na, spec.nj
        self.q_idx_t = torch.tensor(spec.q_idx, dtype=torch.long, device=self.dev)
        self.qd_idx_t = torch.tensor(spec.qd_idx, dtype=torch.long, device=self.dev)
        self.act_leg_t = torch.tensor(spec.act_leg_idx, dtype=torch.long, device=self.dev)
        self.act_wheel_t = torch.tensor(spec.act_wheel_idx, dtype=torch.long, device=self.dev)
        self.spring_t = torch.tensor(spec.spring_idx, dtype=torch.long, device=self.dev)
        # q / qd 在状态向量里的局部位置
        self.q_pos = {int(j): k for k, j in enumerate(spec.q_idx)}
        self.qd_pos = {int(j): k for k, j in enumerate(spec.qd_idx)}
        self.leg_act_in_q = [self.q_pos[j] for j in spec.act_leg_idx if j in self.q_pos]
        self.wheel_in_qd = [self.qd_pos[j] for j in spec.wheel_idx]
        self.leg_act_in_qd = [self.qd_pos[j] for j in spec.act_leg_idx]
        self.default_q = default_q if default_q is not None else torch.zeros(self.nj, device=self.dev)
        # 观测噪声/尺度（按组，参考 Table A.1 与 wheeled-legged_RL 的 self_obs_noise_cfg）
        noise = torch.zeros(self.nq + self.nd + 6 + self.na, device=self.dev)
        noise[: self.nq] = 0.01
        noise[self.nq : self.nq + self.nd] = 0.5
        noise[self.nq + self.nd : self.nq + self.nd + 3] = 0.25
        noise[self.nq + self.nd + 3 : self.nq + self.nd + 6] = 0.05
        scale = torch.ones_like(noise)
        scale[self.nq : self.nq + self.nd] = 0.1
        for k in self.wheel_in_qd:  # 轮速量级大，单独压一档
            noise[self.nq + k] = 1.0
            scale[self.nq + k] = 0.05
        self.obs_noise, self.obs_scale = noise, scale
        self.max_step_reward = MAX_STEP_REWARD
        # 想象 rollout 的状态夹紧范围（防止世界模型长程发散）
        if spec.q_lo is not None and spec.q_hi is not None:
            self.q_lo_t = spec.q_lo[self.q_idx_t]
            self.q_hi_t = spec.q_hi[self.q_idx_t]
        else:
            self.q_lo_t = self.q_hi_t = None
        if qd_limit is not None:
            qd_limit = qd_limit if torch.is_tensor(qd_limit) else torch.tensor(qd_limit, device=self.dev)
            self.qd_max = qd_limit[self.qd_idx_t].abs()
        else:
            self.qd_max = torch.full((self.nd,), 80.0, device=self.dev)

    # 维度
    @property
    def state_dim(self) -> int:
        return 13 + self.nq + self.nd

    @property
    def wm_out_dim(self) -> int:
        return 12 + self.nq + self.nd

    @property
    def inv_dim(self) -> int:
        return 13 + self.nq + self.nd

    @property
    def obs_dim(self) -> int:
        return self.nq + self.nd + 6 + self.na

    @property
    def critic_extra(self) -> int:
        return 4 + self.xi_dim

    # 状态编解码
    def flatten_state(self, S) -> torch.Tensor:
        from .rotations import mat_to_quat

        p, R, v, w, q, qd = S
        return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)

    def unflatten_state(self, x: torch.Tensor):
        from .rotations import quat_to_mat

        nq, nd = self.nq, self.nd
        p = x[..., 0:3]
        R = quat_to_mat(x[..., 3:7])
        v, w = x[..., 7:10], x[..., 10:13]
        q = x[..., 13 : 13 + nq]
        qd = x[..., 13 + nq : 13 + nq + nd]
        return (p, R, v, w, q, qd)

    def full_joints(self, q, qd):
        """把 (nq, nd) 拼回 articulation 的完整关节向量（轮角、云台角填 0）。"""
        B = q.shape[0]
        qf = torch.zeros(B, self.nj, device=q.device, dtype=q.dtype)
        qdf = torch.zeros(B, self.nj, device=q.device, dtype=q.dtype)
        qf.index_copy_(1, self.q_idx_t, q)
        qdf.index_copy_(1, self.qd_idx_t, qd)
        return qf, qdf

    def sanitize(self, S):
        """把想象出来的状态夹到物理合理范围内。"""
        p, R, v, w, q, qd = S
        p = torch.stack((p[..., 0], p[..., 1], p[..., 2].clamp(0.02, 2.0)), -1)
        v = v.clamp(-20.0, 20.0)
        w = w.clamp(-40.0, 40.0)
        if self.q_lo_t is not None:
            q = torch.maximum(torch.minimum(q, self.q_hi_t), self.q_lo_t)
        qd = torch.maximum(torch.minimum(qd, self.qd_max), -self.qd_max)
        return (p, R, v, w, q, qd)

    def invariant_features(self, S) -> torch.Tensor:
        return invariant_features(S)

    # 观测
    def obs(self, S, a_prev) -> torch.Tensor:
        p, R, v, w, q, qd = S
        Rt = R.transpose(-1, -2)
        q0 = self.default_q[self.q_idx_t]
        g_b = -R[..., 2, :]
        return torch.cat((q - q0, qd, (Rt @ w.unsqueeze(-1)).squeeze(-1), g_b, a_prev), -1)

    def add_noise(self, o: torch.Tensor) -> torch.Tensor:
        return o + (2 * torch.rand_like(o) - 1) * self.obs_noise

    # 奖励
    def _wheel_pos_body(self, S):
        if self.kin is None:
            return None
        p, R, v, w, q, qd = S
        qf, qdf = self.full_joints(q, qd)
        bp = self.kin.forward(p, R, v, w, qf, qdf)[0]  # (B, K, 3)
        ids = [self.kin.tracked_names.index(n) for n in self.kin.wheel_names if n in self.kin.tracked_names]
        if len(ids) < 2:
            return None
        rel = bp[:, ids] - p.unsqueeze(1)
        return (R.transpose(-1, -2).unsqueeze(1) @ rel.unsqueeze(-1)).squeeze(-1)

    def reward(self, S, f, a, a_prev, terms: bool = False):
        c = self.cfg
        p, R, v, w, q, qd = S
        cmd = self.cmd.cmd_at(f)
        vx_cmd, wz_cmd, h_cmd = cmd[..., 0], cmd[..., 2], cmd[..., 3]
        Rt = R.transpose(-1, -2)
        v_b = (Rt @ v.unsqueeze(-1)).squeeze(-1)
        w_b = (Rt @ w.unsqueeze(-1)).squeeze(-1)
        # 水平前向速度：机体 x 轴投影到水平面后与速度点乘（上坡时不虚高）
        fx, fy = R[..., 0, 0], R[..., 1, 0]
        n = torch.sqrt(fx * fx + fy * fy).clamp(min=1e-6)
        v_fwd = (v[..., 0] * fx + v[..., 1] * fy) / n
        v_lat = (v[..., 1] * fx - v[..., 0] * fy) / n  # 侧向（cmd_y 恒 0 时也算正则项）
        g_b = -R[..., 2, :]
        roll, pitch = g_b[..., 0], g_b[..., 1]

        e_lin = torch.clamp(vx_cmd - v_fwd, -c.lin_err_clip, c.lin_err_clip)
        e_lat = torch.clamp(cmd[..., 1] - v_lat, -c.lin_err_clip, c.lin_err_clip)
        e_ang = torch.clamp(wz_cmd - w_b[..., 2], -c.ang_err_clip, c.ang_err_clip)
        e_h = torch.clamp(h_cmd - p[..., 2], -c.height_err_clip, c.height_err_clip)

        r_track_lin = c.w_track_lin * torch.exp(-(e_lin**2 + 0.5 * e_lat**2) / c.sigma_lin)
        r_track_lin_sq = c.w_track_lin_sq * (c.sigma_lin_sq * e_lin) ** 2
        r_track_ang = c.w_track_ang * torch.exp(-(e_ang**2) / c.sigma_ang)
        r_track_ang_sq = c.w_track_ang_sq * (c.sigma_ang_sq * e_ang) ** 2
        r_height = c.w_height * torch.exp(-(e_h**2) / c.sigma_height)
        r_height_sq = c.w_height_sq * (c.sigma_height_sq * e_h) ** 2

        r_roll = c.w_roll_exp * torch.exp(-(roll**2) / c.sigma_roll)
        r_pitch = c.w_pitch_exp * torch.exp(-(pitch**2) / c.sigma_pitch)
        amp = c.orient_v_A * torch.exp(-(vx_cmd**2) / c.orient_v_sigma) + c.orient_v_bias
        r_roll_v = c.w_roll_v * (amp * roll) ** 2
        r_pitch_v = c.w_pitch_v * (amp * pitch) ** 2

        r_vz = c.w_lin_vel_z * v_b[..., 2] ** 2
        r_wxy = c.w_ang_vel_xy * (w_b[..., 0] ** 2 + w_b[..., 1] ** 2)
        r_rate = c.w_action_rate * ((a - a_prev) ** 2).sum(-1)

        a_leg = a[:, : len(self.spec.act_leg_idx)]
        a_wheel = a[:, len(self.spec.act_leg_idx) :]
        qd_leg = qd[:, self.leg_act_in_qd]
        qd_wheel = qd[:, self.wheel_in_qd]
        q_leg = q[:, self.leg_act_in_q]
        r_lvel = c.w_leg_vel * (qd_leg**2).sum(-1)
        r_wvel = c.w_wheel_vel * (qd_wheel**2).sum(-1)
        # 力矩估计：tau = kp*(q*-q) - kd*qd （与 ImplicitActuator 一致的隐式 PD）
        q_leg_tgt = self.default_q[self.act_leg_t] + ACTION_SCALE_LEG * a_leg
        v_wheel_tgt = torch.clamp(ACTION_SCALE_WHEEL * a_wheel, -WHEEL_VEL_LIMIT, WHEEL_VEL_LIMIT)
        tau_leg = KP_LEG * (q_leg_tgt - q_leg) - KD_LEG * qd_leg
        tau_leg = torch.clamp(tau_leg, -EFF_LEG, EFF_LEG)
        tau_wheel = torch.clamp(KD_WHEEL * (v_wheel_tgt - qd_wheel), -EFF_WHEEL, EFF_WHEEL)
        r_torque = c.w_joint_torque * (tau_leg**2).sum(-1)
        r_wpower = c.w_wheel_power * torch.clamp(tau_wheel * qd_wheel, min=0.0).sum(-1)

        wpb = self._wheel_pos_body(S)
        if wpb is not None:
            dx = wpb[:, 0, 0] - wpb[:, 1, 0]
            over = torch.clamp(dx.abs() - c.no_fork_distance, min=0.0)
            r_fork = c.w_no_fork * (c.no_fork_sigma * over) ** 2
        else:
            dx = torch.zeros_like(v_fwd)
            r_fork = torch.zeros_like(v_fwd)

        term = self.termination(S).to(v_fwd.dtype)
        r_term = c.w_termination * term

        r = (
            r_track_lin + r_track_lin_sq + r_track_ang + r_track_ang_sq + r_height + r_height_sq
            + r_roll + r_pitch + r_roll_v + r_pitch_v + r_vz + r_wxy + r_rate
            + r_lvel + r_wvel + r_torque + r_wpower + r_fork + r_term
        ) * CONTROL_DT  # fmt: skip
        if not terms:
            return r
        return r, {
            # 正项（上限见 RewardCfg 的 w_*）
            "track_lin": r_track_lin * CONTROL_DT,
            "track_ang": r_track_ang * CONTROL_DT,
            "track_height": r_height * CONTROL_DT,
            "roll_exp": r_roll * CONTROL_DT,
            "pitch_exp": r_pitch * CONTROL_DT,
            # 罚项（≤0）
            "track_lin_sq": r_track_lin_sq * CONTROL_DT,
            "track_ang_sq": r_track_ang_sq * CONTROL_DT,
            "height_sq": r_height_sq * CONTROL_DT,
            "roll_v": r_roll_v * CONTROL_DT,
            "pitch_v": r_pitch_v * CONTROL_DT,
            "lin_vel_z": r_vz * CONTROL_DT,
            "ang_vel_xy": r_wxy * CONTROL_DT,
            "action_rate": r_rate * CONTROL_DT,
            "leg_vel": r_lvel * CONTROL_DT,
            "wheel_vel": r_wvel * CONTROL_DT,
            "torque": r_torque * CONTROL_DT,
            "wheel_power": r_wpower * CONTROL_DT,
            "no_fork": r_fork * CONTROL_DT,
            "termination": r_term * CONTROL_DT,
            # 诊断量
            "e_lin": e_lin.detach(),
            "e_ang": e_ang.detach(),
            "e_height": e_h.detach(),
            "fork_dx": dx.detach(),
            "tau_leg": tau_leg.detach().abs().max(-1).values,
            "terminated": term.detach(),
        }

    # 终止
    def termination(self, S) -> torch.Tensor:
        p, R, v, w, q, qd = S
        too_low = p[..., 2] < self.cfg.term_height
        tilted = R[..., 2, 2] < self.cfg.term_tilt_cos
        fast = v.norm(dim=-1) > self.cfg.term_speed
        bad = ~torch.isfinite(q).all(-1) | ~torch.isfinite(qd).all(-1)
        return too_low | tilted | fast | bad

    def soft_termination(self, S, tz: float = 0.01, tt: float = 0.02) -> torch.Tensor:
        p, R = S[0], S[1]
        low = torch.sigmoid((self.cfg.term_height - p[..., 2]) / tz)
        tilt = torch.sigmoid((self.cfg.term_tilt_cos - R[..., 2, 2]) / tt)
        return 1 - (1 - low) * (1 - tilt)

    # 评论家
    def critic_features(self, S, f, xi_n) -> torch.Tensor:
        p, R, v, w, q, qd = S
        cmd = self.cmd.cmd_at(f)
        Rt = R.transpose(-1, -2)
        v_b = (Rt @ v.unsqueeze(-1)).squeeze(-1)
        w_b = (Rt @ w.unsqueeze(-1)).squeeze(-1)
        err = torch.stack(
            (v_b[..., 0] - cmd[..., 0], v_b[..., 1] - cmd[..., 1], w_b[..., 2] - cmd[..., 2], p[..., 2] - cmd[..., 3]),
            -1,
        )
        return torch.cat((self.invariant_features(S), err, xi_n), -1)


def normalize_xi(xi: torch.Tensor) -> torch.Tensor:
    """ξ = [mu(1), spring_bias(1), com(3)] -> 归一化后的世界模型条件向量。"""
    return torch.cat(((xi[..., :1] - 0.75) / 0.45, xi[..., 1:2] / SPRING_BIAS_RANGE, xi[..., 2:5] / 0.05), -1)
