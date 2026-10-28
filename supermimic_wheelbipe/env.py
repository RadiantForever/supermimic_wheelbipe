# SPDX-License-Identifier: MIT
"""批量仿真器（Isaac Sim / Isaac Lab 2.x + PhysX）：施加动作 -> 步进 -> 用解析奖励打标签。"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from .mdp import MAX_STEP_REWARD
from .robot import (
    ACTION_CLIP,
    ACTION_SCALE_LEG,
    ACTION_SCALE_WHEEL,
    CONTROL_DT,
    DECIMATION,
    SIM_DT,
    SPRING_BIAS_RANGE,
    WHEEL_RADIUS,
    WHEEL_VEL_LIMIT,
    RobotSpec,
    spring_force,
)
from .rotations import mat_to_quat, quat_to_mat, so3_exp


def _t(v):
    """兼容 Isaac Lab 2.x（直接是 Tensor）与 3.x（.torch 包装）。"""
    return v.torch if hasattr(v, "torch") else v


def _replace_cfg(obj, **kw):
    """浅拷贝 cfg 并改字段（Isaac Lab 2.3 的 isaaclab.utils 里没有 replace）。"""
    import copy

    out = copy.copy(obj)
    for k, v in kw.items():
        setattr(out, k, v)
    return out


# 场景 / 物理
def make_material(friction: float = 1.0) -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=friction,
        dynamic_friction=friction,
        restitution=0.0,
    )


def make_sim_cfg(device: str, friction: float = 1.0) -> sim_utils.SimulationCfg:
    """与原项目 ``wheelbipe25_v3/env_cfg.py`` 的 SimulationCfg 完全一致。"""
    return sim_utils.SimulationCfg(
        dt=SIM_DT,
        render_interval=DECIMATION,
        device=device,
        physics_material=make_material(friction),
    )


def make_scene_cfg(num_envs: int, robot_cfg, env_spacing: float = 2.0, friction: float = 1.0):
    @configclass
    class _Cfg(InteractiveSceneCfg):
        # 地面材质要和原项目一致（乘性混合，μ=1.0），否则轮子会打滑
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(physics_material=make_material(friction)),
        )
        # 灯光：USD 里没有光源，缺了它就是一片黑（录视频/看画面时必须要有）
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.92, 0.92, 0.95)),
        )

        robot = _replace_cfg(robot_cfg, prim_path="{ENV_REGEX_NS}/Robot")

    kwargs = dict(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=True)
    try:
        return _Cfg(**kwargs, filter_collisions=True)
    except TypeError:
        return _Cfg(**kwargs)


def make_world(sim_cfg, num_envs: int, robot_cfg, env_spacing: float = 2.0):
    """先建 sim+scene，再从中读出关节名，才能建 spec / MDP / 世界模型。"""
    mat = getattr(sim_cfg, "physics_material", None)
    fr = float(getattr(mat, "static_friction", 1.0) or 1.0)
    # 关键：机器人也要显式绑定材质。USD 自带材质 μ 可能是 0，而 combine="multiply" 时
    # 0 × 任何值 = 0 —— 轮子会空转到 58 rad/s 而车身不动的"无摩擦"现象就是这么来的。
    try:
        robot_cfg.spawn.physics_material = make_material(fr)
    except Exception:  # noqa: BLE001
        pass
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(make_scene_cfg(num_envs, robot_cfg, env_spacing, fr))
    sim.reset()
    return sim, scene, scene["robot"]


# 站立高度标定
@torch.no_grad()
def measure_stand_height(sim, scene, robot, spec, z_init: float, steps: int = 120, tail: int = 40,
                         spring_bias: float = 0.0, verbose: bool = True) -> tuple[float, float, torch.Tensor]:
    """零动作静置若干步，测出机器人**真实的稳态站立高度**。"""
    N = int(getattr(robot, "num_instances", 1))
    dev = str(getattr(robot, "device", "cuda:0"))
    org = scene.env_origins
    q = torch.zeros(N, spec.nj, device=dev)
    qd = torch.zeros_like(q)
    pose = torch.zeros(N, 7, device=dev)
    pose[:, :3] = org
    pose[:, 2] = org[:, 2] + z_init
    pose[:, 3] = 1.0
    for name in ("write_root_link_pose_to_sim", "write_root_pose_to_sim"):
        fn = getattr(robot, name, None)
        if fn is not None:
            fn(pose, env_ids=None)
            break
    for name in ("write_root_link_velocity_to_sim", "write_root_velocity_to_sim"):
        fn = getattr(robot, name, None)
        if fn is not None:
            fn(torch.zeros(N, 6, device=dev), env_ids=None)
            break
    robot.write_joint_state_to_sim(q, qd, None, None)
    scene.update(0.0)

    leg_t = torch.tensor(spec.act_leg_idx, dtype=torch.long, device=dev)
    wheel_t = torch.tensor(spec.act_wheel_idx, dtype=torch.long, device=dev)
    spring_t = torch.tensor(spec.spring_idx, dtype=torch.long, device=dev)
    zs, cos_t = [], []
    for i in range(steps):
        robot.set_joint_position_target(torch.zeros(N, len(leg_t), device=dev), joint_ids=spec.act_leg_idx)
        robot.set_joint_velocity_target(torch.zeros(N, len(wheel_t), device=dev), joint_ids=spec.act_wheel_idx)
        robot.set_joint_position_target(_t(robot.data.joint_pos)[:, wheel_t], joint_ids=spec.act_wheel_idx)
        if len(spec.spring_idx) > 0:
            sp = _t(robot.data.joint_pos)[:, spring_t]
            b = torch.full((N, 1), float(spring_bias), device=dev)
            robot.set_joint_effort_target(spring_force(sp, b), joint_ids=spec.spring_idx)
        scene.write_data_to_sim()
        for _ in range(DECIMATION):
            sim.step(render=False)
        scene.update(CONTROL_DT)
        if i >= steps - tail:
            d = robot.data
            pose = _t(d.root_link_pose_w) if hasattr(d, "root_link_pose_w") else _t(d.root_pose_w)
            zs.append(pose[:, 2] - org[:, 2])
            cos_t.append(quat_to_mat(pose[:, 3:7])[:, 2, 2])
    z_t = torch.stack(zs)
    c_t = torch.stack(cos_t)
    z = float(z_t.median(dim=0).values.mean()) if N > 1 else float(z_t.median())
    c = float(c_t.mean())
    q_settled = _t(robot.data.joint_pos)[0].clone()
    if verbose:
        print(f"[settle] 零动作 {steps} 步后：根高度 {z:.3f} m（z_init={z_init:.3f}），倾角 cos≈{c:.3f}")
        if c < 0.9:
            print("[settle] 警告：静置时姿态不水平（cos<0.9），说明零动作下机器人会倒。"
                  "请先用 tools/calibrate.py 看弹簧力/执行器是否正常。")
    return z, c, q_settled


# solve_stance / _fk_wheels 已移到 kinematics.py，这里只 re-export
from .kinematics import solve_stance  # noqa: E402,F401


def resolve_stand_height(sim, scene, robot, spec, kin, default_q, settle_steps: int = 120,
                         override: float | None = None, verbose: bool = True) -> tuple[float, float]:
    """统一入口：返回 (用于重置/指令高度的稳态站立高度, FK 估计的触地高度)。"""
    z_init = 0.38
    if kin is not None:
        try:
            z_init = kin.rest_height(default_q, WHEEL_RADIUS, kin.wheel_names)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"[stand] FK 估计触地高度失败（{e}），用 0.38 m")
    if override is not None:
        return float(override), z_init
    if settle_steps > 0:
        z, _, _ = measure_stand_height(sim, scene, robot, spec, z_init, steps=settle_steps, verbose=verbose)
        return float(z), z_init
    return float(z_init), z_init


# 环境
class TrackingSim:
    def __init__(self, sim, scene, robot, spec: RobotSpec, cmd, mdp,
                 gamma: float = 0.99, eval_mode: bool = False, max_episode_steps: int = 500,
                 reset_noise: bool = True, z_stand: float | None = None, friction: float = 1.0,
                 com_random: tuple = (0.02, 0.03, 0.03), spring_bias: float | None = None,
                 verbose: bool = True):
        self.sim, self.scene, self.robot = sim, scene, robot
        self.spec, self.cmd, self.mdp, self.gamma = spec, cmd, mdp, gamma
        self.device = cmd.cmd.device
        self.N = int(scene.cfg.num_envs)
        self.eval_mode = eval_mode
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        dev = self.device

        if list(robot.joint_names) != list(spec.joint_names):
            raise RuntimeError(
                "关节顺序与 spec 不一致（说明 USD 与建 spec 时用的不是同一个）：\n"
                f"sim : {list(robot.joint_names)}\nspec: {list(spec.joint_names)}"
            )

        self.default_q = _t(robot.data.default_joint_pos)[0].clone()
        self.leg_ids = list(spec.act_leg_idx)
        self.wheel_ids = list(spec.act_wheel_idx)
        self.spring_ids = list(spec.spring_idx)
        self.leg_t = torch.tensor(self.leg_ids, dtype=torch.long, device=dev)
        self.wheel_t = torch.tensor(self.wheel_ids, dtype=torch.long, device=dev)
        self.spring_t = torch.tensor(self.spring_ids, dtype=torch.long, device=dev)
        self.default_leg = self.default_q[self.leg_t]
        self.nl = len(self.leg_ids)
        self.spring_bias_range = float(SPRING_BIAS_RANGE if spring_bias is None else spring_bias)
        self.render_physics = False  # play 时置 True 才会把画面画到视口

        # 站立姿态：全零关节角 = 腿完全折叠、机身趴到 0.177 m，老师策略没见过这种状态会直接前倾扑倒。
        # 原项目用 IK 蹲姿 + root 0.25 复位；这里用可微 FK 反解一个"轮子触地、轮心在机身下方"的姿态。
        self.stance_q = self.default_q.clone()
        _kin = getattr(mdp, "kin", None)
        if _kin is not None and len(spec.act_leg_idx) > 0:
            try:
                sq, sh = solve_stance(_kin, spec, self.default_q, target_h=max(float(z_stand) + 0.07, 0.22))
                self.stance_q = sq
                if sh is not None and sh > 0.15:
                    self.z_stand = float(sh)
            except Exception as e:  # noqa: BLE001
                print(f"[env] 站立姿态反解失败（{e}），仍用全零关节角")

        # 站立高度：轮子恰好触地时的根高度
        if z_stand is None:
            z_stand = 0.38
            if mdp.kin is not None:
                try:
                    z_stand = mdp.kin.rest_height(self.default_q, WHEEL_RADIUS, mdp.kin.wheel_names)
                except Exception as e:  # noqa: BLE001
                    print(f"[env] FK 估计站立高度失败（{e}），退回 0.38 m")
        self.z_stand = float(z_stand)

        # sim context ξ = [mu(1), spring_bias(1), com(3)]
        self.xi = torch.zeros(self.N, 5, device=dev)
        self.xi[:, 0] = friction
        self._com_ok = False
        self._setup_com((0.0, 0.0, 0.0) if eval_mode else com_random)

        # bookkeeping
        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(self.N, spec.na, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = torch.full((cmd.num_bins,), 0.0, device=dev)
        self.bin_w_avg = torch.full((cmd.num_bins,), 1.0 / cmd.num_bins, device=dev)
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.reset(torch.arange(self.N, device=dev))
        if verbose:
            print(
                f"[env] N={self.N} z_stand={self.z_stand:.3f} m na={spec.na} nq={spec.nq} nd={spec.nd} "
                f"com_random={'on' if self._com_ok else 'off'}"
            )

    # ξ
    def _setup_com(self, rng):
        if max(rng) <= 0.0:
            return
        dev = self.device
        base_id = self.robot.find_bodies("base_link")[0][0]
        self._base_id = base_id
        gain = torch.stack(
            (
                torch.empty(self.N, device=dev).uniform_(-rng[0], rng[0]),
                torch.empty(self.N, device=dev).uniform_(-rng[1], rng[1]),
                torch.empty(self.N, device=dev).uniform_(-rng[2], rng[2]),
            ),
            -1,
        )
        com0 = _t(self.robot.data.body_com_pose_b)[:, base_id, :3].clone()
        for name, kw in (("set_coms_index", "kw"), ("set_coms", "kw"), ("set_coms_index", "pos"), ("set_coms", "pos")):
            fn = getattr(self.robot, name, None)
            if fn is None:
                continue
            try:
                if kw == "kw":
                    fn(coms=(com0 + gain).unsqueeze(1), body_ids=[base_id])
                else:
                    fn((com0 + gain).unsqueeze(1), [base_id])
                self._com_ok = True
                break
            except Exception:  # noqa: BLE001
                continue
        if self._com_ok:
            self.xi[:, 2:5] = gain
        else:
            print("[env] 该 Isaac Lab 版本没有 set_coms_index/set_coms，CoM 随机化关闭（ξ 的 com 分量置 0）")

    # 状态
    def _root_pose(self, d):
        for name in ("root_link_pose_w", "root_pose_w"):
            v = getattr(d, name, None)
            if v is not None:
                return _t(v)
        raise AttributeError("ArticulationData 里找不到根位姿")

    def _root_vel(self, d):
        for name in ("root_link_vel_w", "root_vel_w"):
            v = getattr(d, name, None)
            if v is not None:
                return _t(v)
        return torch.cat((_t(d.root_lin_vel_w), _t(d.root_ang_vel_w)), -1)

    def state(self):
        d = self.robot.data
        pose = self._root_pose(d)
        vel = self._root_vel(d)
        p = pose[:, :3] - self.scene.env_origins
        R = quat_to_mat(pose[:, 3:7])
        jp, jv = _t(d.joint_pos), _t(d.joint_vel)
        return (p, R, vel[:, :3], vel[:, 3:6], jp[:, self.spec.q_idx], jv[:, self.spec.qd_idx])

    def state_full(self):
        d = self.robot.data
        return _t(d.joint_pos).clone(), _t(d.joint_vel).clone()

    # 重置
    def sample_frames(self, n: int):
        m = self.cmd
        # 原版的 bin_G 恒 ≥0（奖励全是正的），而本任务有负项（终止/正则），
        # 所以这里把 G 夹到 [0, G_max]：回报越差权重越高，但有上界，不会让某个 bin 独占采样。
        soft = torch.softmax(-25.0 * self.bin_G.clamp(min=0.0, max=self.G_max) / self.G_max, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + 50, m.clip_end[m.bin_clip[bins]])
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = self.cmd.frame_bin[frames]
        spec = self.spec

        q = self.stance_q.unsqueeze(0).repeat(n, 1).clone()  # 站立姿态（不是全零）
        qd = torch.zeros(n, spec.nj, device=dev)
        if self.reset_noise:
            q[:, self.leg_t] += torch.empty(n, self.nl, device=dev).uniform_(-0.03, 0.03)
            p = torch.stack(
                (
                    torch.empty(n, device=dev).uniform_(-0.05, 0.05),
                    torch.empty(n, device=dev).uniform_(-0.05, 0.05),
                    torch.empty(n, device=dev).uniform_(-0.01, 0.02),
                ),
                -1,
            )
            rv = torch.stack(
                (
                    torch.empty(n, device=dev).uniform_(-0.03, 0.03),
                    torch.empty(n, device=dev).uniform_(-0.03, 0.03),
                    torch.empty(n, device=dev).uniform_(-3.1416, 3.1416),
                ),
                -1,
            )
            R = so3_exp(rv)
        else:
            p = torch.zeros(n, 3, device=dev)
            R = torch.eye(3, device=dev).expand(n, 3, 3)
        p = torch.stack((p[:, 0], p[:, 1], self.z_stand + p[:, 2]), -1)
        if spec.q_lo is not None and spec.q_hi is not None:
            q = torch.maximum(torch.minimum(q, spec.q_hi), spec.q_lo)

        pose = torch.cat((p + self.scene.env_origins[env_ids], mat_to_quat(R)), -1)
        try:
            self.scene.reset(env_ids)
        except Exception:  # noqa: BLE001
            pass
        self._write_root_pose(pose, env_ids)
        self._write_root_vel(torch.zeros(n, 6, device=dev), env_ids)
        self._write_joints(q, qd, env_ids)
        self.robot.set_joint_position_target(q[:, self.leg_t], joint_ids=self.leg_ids, env_ids=env_ids)
        self.robot.set_joint_velocity_target(qd[:, self.wheel_t], joint_ids=self.wheel_ids, env_ids=env_ids)
        if len(self.spring_ids) > 0:
            self.robot.set_joint_effort_target(qd[:, self.spring_t], joint_ids=self.spring_ids, env_ids=env_ids)
        self.scene.update(0.0)

        self.f[env_ids] = frames
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        self.done_eval[env_ids] = False  # 手动 reset 时也要把评测标记清掉
        self.failed_eval[env_ids] = False
        if not self.eval_mode and len(self.spring_ids) > 0 and self.spring_bias_range > 0:
            r = self.spring_bias_range
            self.xi[env_ids, 1] = torch.empty(n, device=dev).uniform_(-r, r)

    def _write_root_pose(self, pose, env_ids):
        for name in ("write_root_link_pose_to_sim", "write_root_pose_to_sim"):
            fn = getattr(self.robot, name, None)
            if fn is not None:
                return fn(pose, env_ids=env_ids)
        raise AttributeError("找不到 write_root_pose_to_sim / write_root_link_pose_to_sim")

    def _write_root_vel(self, vel, env_ids):
        for name in ("write_root_link_velocity_to_sim", "write_root_velocity_to_sim"):
            fn = getattr(self.robot, name, None)
            if fn is not None:
                return fn(vel, env_ids=env_ids)
        raise AttributeError("找不到 write_root_velocity_to_sim / write_root_link_velocity_to_sim")

    def _write_joints(self, q, qd, env_ids):
        try:
            self.robot.write_joint_state_to_sim(q, qd, None, env_ids)
        except TypeError:
            self.robot.write_joint_state_to_sim(q, qd, env_ids=env_ids)

    def refresh(self):
        self.scene.update(0.0)

    # 动作 / 物理
    def apply_action(self, a: torch.Tensor):
        a = torch.clamp(a, -ACTION_CLIP, ACTION_CLIP)
        q_leg = self.default_leg + ACTION_SCALE_LEG * a[:, : self.nl]
        v_wheel = torch.clamp(ACTION_SCALE_WHEEL * a[:, self.nl :], -WHEEL_VEL_LIMIT, WHEEL_VEL_LIMIT)
        self.robot.set_joint_position_target(q_leg, joint_ids=self.leg_ids)
        self.robot.set_joint_velocity_target(v_wheel, joint_ids=self.wheel_ids)
        cur_wheel = _t(self.robot.data.joint_pos)[:, self.wheel_t]
        self.robot.set_joint_position_target(cur_wheel, joint_ids=self.wheel_ids)
        if len(self.spring_ids) > 0:
            sp = _t(self.robot.data.joint_pos)[:, self.spring_t]
            self.robot.set_joint_effort_target(spring_force(sp, self.xi[:, 1:2]), joint_ids=self.spring_ids)

    def physics(self):
        self.scene.write_data_to_sim()
        for _ in range(DECIMATION):
            self.sim.step(render=self.render_physics)
        self.scene.update(CONTROL_DT)

    # 一步
    def step(self, a: torch.Tensor) -> dict:
        a = torch.clamp(a, -ACTION_CLIP, ACTION_CLIP)  # 与想象里的动作范围一致（buffer 里存的也是夹紧后的）
        S = self.state()
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()

        self.apply_action(a)
        self.physics()

        S1 = self.state()
        f1 = f + 1
        r, terms = self.mdp.reward(S1, f1, a, a_prev, terms=True)
        term = self.mdp.termination(S1)
        clip_end = self.cmd.clip_end[self.cmd.clip_id[f]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        out = {
            "s": self.mdp.flatten_state(S), "a": a.clone(), "a_prev": a_prev, "f": f,
            "s1": self.mdp.flatten_state(S1), "r": r, "term": term, "done": done,
            "ep_step": ep_step, "xi": xi, "terms": terms,
        }  # fmt: skip

        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = a.clone()
        if self.eval_mode:
            self.failed_eval |= term & ~self.done_eval
            self.done_eval |= done
            return out
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            T = self.ep_step[ids].float()
            boot = torch.where(
                term[ids], torch.zeros_like(T), (self.gamma**T) * (self.ep_ret[ids] / T) / (1 - self.gamma)
            )
            G = self.ep_disc_ret[ids] + boot
            b = self.start_bin[ids]
            self.bin_G.index_put_((b,), 0.8 * self.bin_G[b] + 0.2 * G)
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * MAX_STEP_REWARD)
            self.reset(ids)
        return out
