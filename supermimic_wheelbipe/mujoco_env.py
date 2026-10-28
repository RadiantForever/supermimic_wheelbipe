# SPDX-License-Identifier: MIT
"""MuJoCo 版 wheelbipe 环境（CPU 单机器人，接口对齐 ``env.TrackingSim`` 的 state()/step()）。"""

from __future__ import annotations

import os

import mujoco
import numpy as np
import torch

from .constants import (
    ACTION_CLIP,
    ACTION_SCALE_LEG,
    ACTION_SCALE_WHEEL,
    CONTROL_DT,
    SPRING_BIAS_RANGE,
    WHEEL_RADIUS,
    WHEEL_VEL_LIMIT,
    spring_force,
)

DEFAULT_XML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "assets", "mujoco", "wheelbipe_v14_2.xml")


def _quat_from_yaw(yaw: float) -> list[float]:
    h = 0.5 * float(yaw)
    return [float(np.cos(h)), 0.0, 0.0, float(np.sin(h))]


def load_policy_ckpt(path: str, device: str = "cpu"):
    """读 SuperMimic 检查点，返回 (policy, spec, args)。"""
    ck = torch.load(path, map_location=device)
    sp = dict(ck["spec"])
    from .spec import RobotSpec

    spec = RobotSpec(**sp)
    a = dict(ck.get("args") or {})
    K = int(a.get("goal_frames", 10))
    Hh = int(a.get("hist", 10))
    obs_dim = spec.nq + spec.nd + 6 + spec.na
    from .networks import Policy

    pi = Policy(obs_dim, spec.na, 4 * K, hist=Hh).to(device)
    sd = ck.get("pi", ck)
    pi.load_state_dict(sd)
    pi.eval()
    a.setdefault("goal_frames", K)
    a.setdefault("hist", Hh)
    a["iteration"] = int(ck.get("iteration", -1))
    return pi, spec, a


class MujocoWheelbipe:
    """单个 wheelbipe（base_link 是自由刚体，其余 36 个关节按 ckpt 的 ``spec.joint_names`` 排列）。"""

    def __init__(self, spec, xml: str = DEFAULT_XML, z_init: float = 0.38,
                 device: str = "cpu", spring_bias: float | None = None,
                 wheel_kv: float | None = None, friction_mu: float | None = None,
                 condim: int | None = None, timestep: float | None = None):
        if not os.path.exists(xml):
            raise FileNotFoundError(
                f"找不到 MJCF: {xml}\n先跑 `python tools/usd2mjcf.py` 从 USD 生成一份。"
            )
        self.spec = spec
        self.device = device
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        if timestep is not None:  # 扫参用：更小的步长换更硬的闭链/接触
            self.m.opt.timestep = float(timestep)
        if friction_mu is not None:
            self.m.geom_friction[:, 0] = float(friction_mu)
        if condim is not None:
            self.m.geom_condim[:] = int(condim)
        self.timestep = float(self.m.opt.timestep)
        self.decim = max(1, int(round(CONTROL_DT / self.timestep)))
        self.control_dt = self.decim * self.timestep
        self.z_init = z_init
        self._bias = 0.0 if spring_bias is None else float(spring_bias)

        m = self.m
        missing = [n for n in spec.joint_names
                   if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) < 0]
        if missing:
            raise RuntimeError(f"MJCF 缺少关节 {missing}；重新跑 tools/usd2mjcf.py")

        self._jnt = {n: m.joint(n).id for n in spec.joint_names}
        qadr = np.array([m.jnt_qposadr[self._jnt[n]] for n in spec.joint_names], dtype=np.int64)
        dadr = np.array([m.jnt_dofadr[self._jnt[n]] for n in spec.joint_names], dtype=np.int64)
        self.q_adr = qadr[spec.q_idx]      # (nq,) 按状态顺序
        self.qd_adr = dadr[spec.qd_idx]    # (nd,)
        self.qadr_all = qadr               # (nj,)
        self.spring_q = dadr[spec.spring_idx]

        self._act = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
        # 缺 actuator 说明 MJCF 不是本工具生成的，早报错
        for n in spec.act_leg_idx + spec.act_wheel_idx + spec.spring_idx:
            key = spec.joint_names[n] + "_act"
            if key not in self._act:
                raise RuntimeError(f"MJCF 缺少执行器 {key}")
        self.a_leg = np.array([self._act[spec.joint_names[n] + "_act"] for n in spec.act_leg_idx])
        self.a_wheel = np.array([self._act[spec.joint_names[n] + "_act"] for n in spec.act_wheel_idx])
        self.a_spring = np.array([self._act[spec.joint_names[n] + "_act"] for n in spec.spring_idx])
        self.a_gimbal = np.array([i for k, i in self._act.items() if k.startswith("gimbal_")], dtype=np.int64)

        self.base_id = m.body("base_link").id
        self.a_prev = torch.zeros(1, spec.na, device=device)
        self.ctrl = np.zeros(m.nu, dtype=np.float64)
        self._zero_leg = torch.zeros(1, len(spec.act_leg_idx), device=device)

        # 显式伺服响应时间常数 I_eff/kv ≈ 0.4 s，与 Isaac 隐式伺服差一个量级
        if wheel_kv is not None:
            for i in self.a_wheel:
                m.actuator_gainprm[i, 0] = float(wheel_kv)
                m.actuator_biasprm[i, 1] = -float(wheel_kv)
        self.reset()

    # 工具
    def default_q(self) -> np.ndarray:
        """全 0 关节角（与 robot.py 的 init_state 一致）。"""
        return np.zeros(len(self.spec.joint_names), dtype=np.float64)

    def rest_height(self) -> float:
        """当前姿态下"轮子刚好触地"所需的 base_link 高度（用 MuJoCo 自己的 FK，不走 pxr）。"""
        mujoco.mj_forward(self.m, self.d)
        base_z = float(self.d.qpos[2])
        wheel_z = min(float(self.d.body(n).xpos[2]) for n in ("left_wheel_link", "right_wheel_link"))
        return base_z - wheel_z + WHEEL_RADIUS

    # 重置
    def reset(self, q_full: np.ndarray | torch.Tensor | None = None,
              z: float | None = None, yaw: float = 0.0) -> None:
        if q_full is None:
            q = self.default_q()
        elif torch.is_tensor(q_full):
            q = q_full.detach().cpu().double().numpy().reshape(-1)
        else:
            q = np.asarray(q_full, dtype=np.float64).reshape(-1)
        if q.size != len(self.spec.joint_names):
            raise ValueError(f"q_full 长度应为 {len(self.spec.joint_names)}，实际 {q.size}")

        d = self.d
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qpos[0:3] = (0.0, 0.0, self.z_init if z is None else float(z))
        d.qpos[3:7] = _quat_from_yaw(yaw)
        d.qpos[self.qadr_all] = q
        d.ctrl[:] = 0.0
        self.a_prev.zero_()
        mujoco.mj_forward(self.m, d)
        self._write_spring()

    # 感知
    def state(self):
        """返回 (p, R, v, w, q, qd)，和 ``env.TrackingSim.state()`` 同语义；B=1。"""
        d = self.d
        p = torch.tensor(d.qpos[0:3], dtype=torch.float32, device=self.device).unsqueeze(0)
        R = torch.tensor(np.array(d.xmat[self.base_id], dtype=np.float64).reshape(1, 3, 3),
                         dtype=torch.float32, device=self.device)
        v = torch.tensor(d.qvel[0:3], dtype=torch.float32, device=self.device).unsqueeze(0)
        w_b = torch.tensor(d.qvel[3:6], dtype=torch.float32, device=self.device)
        w = (R[0] @ w_b).unsqueeze(0)
        q = torch.tensor(d.qpos[self.q_adr], dtype=torch.float32, device=self.device).unsqueeze(0)
        qd = torch.tensor(d.qvel[self.qd_adr], dtype=torch.float32, device=self.device).unsqueeze(0)
        return (p, R, v, w, q, qd)

    @property
    def base_z(self) -> float:
        return float(self.d.qpos[2])

    @property
    def tilt_cos(self) -> float:
        return float(self.d.xmat[self.base_id].reshape(3, 3)[2, 2])

    @property
    def vel_forward(self) -> float:
        """水平前向速度（机体 x 轴投影到水平面），与 ``mdp.reward`` 里的 v_fwd 同口径。"""
        Rm = self.d.xmat[self.base_id].reshape(3, 3)
        fx, fy = float(Rm[0, 0]), float(Rm[1, 0])
        n = max(float(np.hypot(fx, fy)), 1e-6)
        return float((self.d.qvel[0] * fx + self.d.qvel[1] * fy) / n)

    @property
    def ncon(self) -> int:
        return int(self.d.ncon)

    # 动作 / 物理
    def _write_spring(self):
        q = torch.tensor(self.d.qpos[[self.m.jnt_qposadr[j] for j in self.spec.spring_idx]],
                         dtype=torch.float32).unsqueeze(0)
        f = spring_force(q, torch.tensor([[self._bias]], dtype=torch.float32)).squeeze(0)
        self.ctrl[self.a_spring] = f.numpy()
        self.d.ctrl[:] = self.ctrl

    def set_spring_bias(self, bias: float):
        self._bias = float(np.clip(bias, -SPRING_BIAS_RANGE, SPRING_BIAS_RANGE))

    def apply_action(self, a):
        """a: (1, 6) 或 (6,)，语义与 Isaac 完全一致。"""
        if torch.is_tensor(a):
            arr = a.detach().cpu().double().numpy().reshape(-1)
        else:
            arr = np.asarray(a, dtype=np.float64).reshape(-1)
        arr = np.clip(arr, -ACTION_CLIP, ACTION_CLIP)
        nl = len(self.a_leg)
        self.ctrl[self.a_leg] = ACTION_SCALE_LEG * arr[:nl]
        self.ctrl[self.a_wheel] = np.clip(ACTION_SCALE_WHEEL * arr[nl:], -WHEEL_VEL_LIMIT, WHEEL_VEL_LIMIT)
        if self.a_gimbal.size:
            self.ctrl[self.a_gimbal] = 0.0
        self._write_spring()

    def physics(self):
        for _ in range(self.decim):
            mujoco.mj_step(self.m, self.d)

    def step(self, a=None):
        """一个控制步（20 ms）。返回新的状态。"""
        if a is not None:
            self.apply_action(a)
            if torch.is_tensor(a):
                self.a_prev = a.detach().clamp(-ACTION_CLIP, ACTION_CLIP).to(self.device).reshape(1, -1)
        self.physics()
        return self.state()

    def set_zero_action(self):
        self.ctrl[:] = 0.0
        self.d.ctrl[:] = 0.0
        self._write_spring()

    # 调试
    def describe(self) -> str:
        return (f"[mujoco] nq={self.m.nq} nv={self.m.nv} njnt={self.m.njnt} neq={self.m.neq} "
                f"nu={self.m.nu} ngeom={self.m.ngeom} dt={self.timestep * 1e3:.3f}ms "
                f"decim={self.decim} ctrl_dt={self.control_dt * 1e3:.1f}ms")
