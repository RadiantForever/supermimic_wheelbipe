# SPDX-License-Identifier: MIT
"""SuperMimic 训练主循环（Algorithm A.1）：采集 -> 拟合世界模型 -> 想象 + BPTT 更新策略/评论家。"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="SuperMimic-style training for wheelbipe")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=200_000)
parser.add_argument("--buffer_steps", type=int, default=1000)
parser.add_argument("--utd", type=float, default=0.4, help="每次向量化仿真步的更新次数")
parser.add_argument("--horizon", type=int, default=24, help="想象 rollout 长度 H")
parser.add_argument("--goal_frames", type=int, default=10, help="goal 里前视的指令帧数 K")
parser.add_argument("--hist", type=int, default=10)
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batches", type=int, default=10)
parser.add_argument("--wm_batch", type=int, default=2048)
parser.add_argument("--pi_batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--no_critic", action="store_true")
parser.add_argument("--warmup_steps", type=int, default=64)
parser.add_argument("--wm_only_iters", type=int, default=1000)
parser.add_argument("--lr_decay_iters", type=int, default=120_000)
parser.add_argument("--ckpt_every", type=int, default=1000)
parser.add_argument("--last_every", type=int, default=200,
                    help="每隔多少轮写一次 last.pt（否则中断会丢最多 ckpt_every 轮的进度）")
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--max_episode_steps", type=int, default=500, help="10 s @50Hz")
parser.add_argument("--env_spacing", type=float, default=2.0)
parser.add_argument("--friction", type=float, default=1.0)
parser.add_argument("--com_random", type=float, nargs=3, default=(0.02, 0.03, 0.03), metavar=("X", "Y", "Z"))
parser.add_argument("--spring_bias", type=float, default=None, help="覆盖 SPRING_BIAS_RANGE")
parser.add_argument("--z_stand", type=float, default=None, help="覆盖稳态站立高度（默认自动测）")
parser.add_argument("--settle_steps", type=int, default=120, help="自动测站立高度的静置步数，0=跳过")
parser.add_argument("--height_range", type=float, nargs=2, default=None, help="指令高度范围 (m)")
parser.add_argument("--term_height", type=float, default=None, help="终止高度 (m)，默认 z_stand-0.15")
parser.add_argument("--lin_vel_x", type=float, nargs=2, default=(-2.7, 2.7),
                    help="与预训练 PPO 的 lin_vel_x 范围一致")
parser.add_argument("--focus_lo", type=float, default=None, help="混合采样的聚焦下界 (m/s)")
parser.add_argument("--focus_hi", type=float, default=None, help="混合采样的聚焦上界 (m/s)")
parser.add_argument("--focus_prob", type=float, default=0.4, help="每段指令落在聚焦区间的概率")
parser.add_argument("--ang_vel_z", type=float, nargs=2, default=(-6.2832, 6.2832))
parser.add_argument("--no_gimbal_usd", type=int, default=0)
parser.add_argument("--model_json", type=str, default=os.path.join(REPO, "data", "wheelbipe_v14_2_model.json"))
parser.add_argument("--no_kin", action="store_true", help="跳过 FK（同时关闭 no_fork 奖励）")
parser.add_argument("--num_clips", type=int, default=256)
parser.add_argument("--clip_frames", type=int, default=2000)
parser.add_argument("--log_dir", type=str, default=os.path.join(REPO, "logs", "run"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--init_policy", type=str, default=None, help="行为克隆产出的 bc_init.pt（热启动策略）")
parser.add_argument("--teacher_ckpt", type=str, default=None,
                    help="老师策略目录/权重；给了就按 --teacher_mix 比例用老师的动作采集数据")
parser.add_argument("--teacher_mix", type=float, default=0.5, help="采集时用老师动作的 env 比例")
parser.add_argument("--teacher_mix_iters", type=int, default=3000, help="只在前 N 轮混老师数据")
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"])
parser.add_argument("--compile", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import supermimic_wheelbipe.networks as networks  # noqa: E402
from supermimic_wheelbipe.buffer import ReplayBuffer  # noqa: E402
from supermimic_wheelbipe.command import CommandLib  # noqa: E402
from supermimic_wheelbipe.constants import default_term_height, height_window  # noqa: E402
from supermimic_wheelbipe.env import TrackingSim, make_sim_cfg, make_world, resolve_stand_height  # noqa: E402
from supermimic_wheelbipe.kinematics import Kinematics  # noqa: E402
from supermimic_wheelbipe.mdp import MAX_STEP_REWARD, MDP, RewardCfg, normalize_xi  # noqa: E402
from supermimic_wheelbipe.model import ensure_model, wheel_bodies  # noqa: E402
from supermimic_wheelbipe.networks import Critic, Policy, WorldModel  # noqa: E402
from supermimic_wheelbipe.robot import (  # noqa: E402
    ACTION_CLIP,
    WHEEL_RADIUS,
    RobotSpec,
    build_spec,
    make_robot_cfg,
    usd_path,
)


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    dev = args.device
    if dev == "cpu":
        print("[train] 警告：训练需要 GPU，--device cpu 只能用来调试流程")
    os.makedirs(args.log_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.log_dir, "args.json"), "w"), indent=2, default=str)

    # sim + robot
    usd = usd_path(no_gimbal=bool(args.no_gimbal_usd))
    print(f"[train] USD = {usd}")
    sim_cfg = make_sim_cfg(dev, friction=args.friction)
    sim, scene, robot = make_world(sim_cfg, args.num_envs, make_robot_cfg(usd), env_spacing=args.env_spacing)
    print("[train] robot joints:", list(robot.joint_names))

    lim = robot.data.joint_pos_limits
    lim = lim.torch if hasattr(lim, "torch") else lim
    q_lo = lim[0, :, 0].clone()
    q_hi = lim[0, :, 1].clone()
    spec: RobotSpec = build_spec(robot.joint_names, q_lo=q_lo, q_hi=q_hi)
    print(spec.describe())

    # 运动学树（可选）
    kin = None
    if not args.no_kin:
        try:
            model = ensure_model(usd, path=args.model_json)
            wheel_names = wheel_bodies(model, spec)
            kin = Kinematics(
                model, list(robot.joint_names), dev, tracked=list(robot.body_names), wheel_bodies=wheel_names
            )
            miss = getattr(kin, "missing_joints", [])
            if miss:
                print(f"[train] 注意: 运动学树里这些关节在仿真中没有对应项 -> 按固定处理: {miss}")
            print(f"[train] FK: {kin.num_bodies} 刚体 / {kin.num_movable} 可动关节, wheel_bodies={wheel_names}")
        except Exception as e:  # noqa: BLE001
            print(f"[train] FK 不可用（{e}）；no_fork 奖励将关闭，其余照常训练")
            kin = None

    # 站立高度 -> 指令高度范围
    default_q = robot.data.default_joint_pos
    default_q = (default_q.torch if hasattr(default_q, "torch") else default_q)[0].clone()
    # 关键一步：全 0 关节角不是站立姿态，静置一小段取真实稳态高度作为基准
    z_stand, z_init = resolve_stand_height(
        sim, scene, robot, spec, kin, default_q, settle_steps=args.settle_steps, override=args.z_stand
    )
    if args.height_range is not None:
        h_range = (float(args.height_range[0]), float(args.height_range[1]))
    else:
        h_range = height_window(z_stand)
    term_height = float(args.term_height) if args.term_height is not None else default_term_height(z_stand)
    print(f"[train] z_stand={z_stand:.3f} m  height_cmd∈{h_range}  term_height={term_height:.3f}")

    # 指令库 / MDP
    focus = None
    if args.focus_lo is not None and args.focus_hi is not None:
        focus = (float(args.focus_lo), float(args.focus_hi))
        print(f"[train] 混合采样: vx∈{tuple(args.lin_vel_x)} 全范围，"
              f"{args.focus_prob:.0%} 的段落聚焦到 {focus}")
    cmd = CommandLib.random(
        dev, num_clips=args.num_clips, clip_frames=args.clip_frames, seed=args.seed,
        height=h_range, lin_vel_x=tuple(args.lin_vel_x), ang_vel_z=tuple(args.ang_vel_z),
        focus=focus, focus_prob=float(args.focus_prob),
    )  # fmt: skip
    qd_lim = robot.data.joint_vel_limits
    qd_lim = qd_lim.torch if hasattr(qd_lim, "torch") else qd_lim
    mdp = MDP(spec, cmd, kin=kin, default_q=default_q, qd_limit=qd_lim[0] * 2.0,
              cfg=RewardCfg(term_height=term_height))
    env = TrackingSim(
        sim, scene, robot, spec, cmd, mdp,
        gamma=args.gamma, max_episode_steps=args.max_episode_steps,
        z_stand=z_stand, friction=args.friction, com_random=tuple(args.com_random),
        spring_bias=args.spring_bias,
    )  # fmt: skip

    # 网络
    K, Hh, HZ = args.goal_frames, args.hist, args.horizon
    GOAL_DIM = 4 * K
    CRITIC_IN = mdp.inv_dim + mdp.critic_extra + GOAL_DIM
    wm = WorldModel(mdp.inv_dim, spec.na, mdp.xi_dim, mdp.wm_out_dim).to(dev)
    pi = Policy(mdp.obs_dim, spec.na, GOAL_DIM, hist=Hh).to(dev)
    critic = Critic(CRITIC_IN).to(dev)
    opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
    opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
    start_it, wall_offset, sim_steps = 0, 0.0, 0
    if args.resume and not os.path.exists(args.resume):
        raise SystemExit(
            f"[train] --resume 指向的文件不存在: {args.resume}\n"
            "（不静默从零开始，避免白白训练。哪个目录有 last.pt 就用哪个）"
        )
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        wm.load_state_dict(ck["wm"]), pi.load_state_dict(ck["pi"]), critic.load_state_dict(ck["critic"])
        opt_wm.load_state_dict(ck["opt_wm"]), opt_pi.load_state_dict(ck["opt_pi"]), opt_v.load_state_dict(ck["opt_v"])
        start_it, wall_offset, sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
        print(f"[train] resume from {args.resume} @ iteration {start_it}")
    elif args.init_policy and os.path.exists(args.init_policy):
        # 行为克隆热启动：只装策略（含 obs/goal 的运行归一化统计），世界模型/评论家仍从零学
        bc = torch.load(args.init_policy, map_location=dev)
        meta = bc.get("args", {})
        want = {"obs_dim": mdp.obs_dim, "na": spec.na, "goal_frames": K, "hist": Hh}
        bad = {k: (meta.get(k), v) for k, v in want.items() if k in meta and meta[k] != v}
        if bad:
            raise SystemExit(f"--init_policy 与当前超参不一致（键: 权重值, 当前值）: {bad}")
        pi.load_state_dict(bc["pi"])
        print(f"[train] init policy from {args.init_policy}（行为克隆热启动）")

    # 老师策略（可选）
    teacher = None
    if args.teacher_ckpt:
        from supermimic_wheelbipe.teacher import Teacher, pick_best_teacher

        try:
            teacher = Teacher(spec, list(robot.joint_names), default_q,
                              pick_best_teacher(args.teacher_ckpt), dev)  # fmt: skip
            print(f"[train] 老师已加载：前 {args.teacher_mix_iters} 轮按 {args.teacher_mix:.0%} "
                  f"比例用老师动作采集数据")
        except Exception as e:  # noqa: BLE001
            print(f"[train] 老师加载失败（{e}），只用策略自己的数据继续训练")
            teacher = None

    err_floor = torch.cat(
        (
            torch.full((3,), 3e-3),
            torch.full((3,), 5e-3),
            torch.full((3,), 2e-2),
            torch.full((3,), 5e-2),
            torch.full((spec.nq,), 3e-3),
            torch.full((spec.nd,), 5e-2),
        )
    ).to(dev)
    err_floor = err_floor[: wm.out_dim]

    # 热函数
    def _wm_rollout(s, a, xi):
        k = a.shape[1]  # 用实际窗口长度：buffer 还没攒够时 sample_windows 会自动缩短
        S = mdp.unflatten_state(s[:, 0])
        preds = []
        for j in range(k):
            S = wm.step(S, a[:, j], normalize_xi(xi[:, j]))
            preds.append(S)
        return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # fmt: skip

    def _wm_err(S_hat, s1, mask):
        std = torch.maximum(wm.out_norm.std(), err_floor)
        e = (wm.state_error(S_hat, mdp.unflatten_state(s1)) / std) ** 2
        e = e.mean(-1)
        return (e * mask).sum() / s1.shape[0], e[:, -1]

    def _imag_step(S, o_hist, a_prev, f, xi_n, g):
        a = torch.clamp(pi(o_hist, g), -ACTION_CLIP, ACTION_CLIP)
        S1 = mdp.sanitize(wm.step(S, a, xi_n))
        o1 = mdp.add_noise(mdp.obs(S1, a))
        o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
        return S1, a, o_hist, f + 1, cmd.goal(f + 1, K)

    def _reward_term(S, f, a, a_prev):
        return mdp.reward(S, f, a, a_prev), mdp.soft_termination(S)

    def _critic_input(S, f, xi_n, g):
        return torch.cat((mdp.critic_features(S, f, xi_n), g), -1)

    if args.compile:
        _wm_rollout = torch.compile(_wm_rollout)
        _wm_err = torch.compile(_wm_err)
        _imag_step = torch.compile(_imag_step)
        _reward_term = torch.compile(_reward_term)
        _critic_input = torch.compile(_critic_input)

    def lr_at(it: int) -> float:
        if it >= args.lr_decay_iters:
            return args.lr * 0.03
        return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))

    def set_requires_grad(m: nn.Module, flag: bool):
        for p in m.parameters():
            p.requires_grad_(flag)

    # 更新
    def wm_update():
        if buffer.size < 3:  # resume 后 buffer 是空的，先攒数据
            return torch.zeros((), device=dev), torch.zeros((), device=dev)
        w = buffer.sample_windows(args.wm_batch, args.wm_window)
        S_hat = _wm_rollout(w["s"], w["a"], w["xi"])
        loss, e_last = _wm_err(S_hat, w["s1"], w["mask"])
        opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        opt_wm.step()
        m = w["mask"][:, -1]
        return loss.detach() / args.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1)

    def sample_valid_starts(B: int):
        starts, hists, n = [], [], 0
        while n < B:
            s, h = buffer.sample_starts(2 * B, Hh)
            f = s["f"]
            ok = f + HZ <= cmd.clip_end[cmd.clip_id[f]]
            idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
            starts.append({k: v[idx] for k, v in s.items()})
            hists.append({k: v[idx] for k, v in h.items()})
            n += len(idx)
            if len(idx) == 0:
                break
        cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
        return cat(starts), cat(hists)

    def policy_update():
        if buffer.size < 2:  # 刚 resume / utd 较大时第一轮可能没采数据
            z = torch.zeros((), device=dev)
            return z, z, z, z, z
        s, h = sample_valid_starts(args.pi_batch)
        S = mdp.unflatten_state(s["s"])
        xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
        xi_n = normalize_xi(xi)
        with torch.no_grad():
            o_hist = mdp.obs(mdp.unflatten_state(h["s"]), h["a_prev"])
        o_hist = mdp.add_noise(o_hist)
        set_requires_grad(wm, False)
        set_requires_grad(critic, False)
        B = s["s"].shape[0]
        g0 = cmd.goal(f, K)
        g, S0 = g0, S
        Ss, As, Aps, Fs, Gs = [], [], [], [], []
        for _ in range(HZ):
            Aps.append(a_prev)
            S, a_prev, o_hist, f, g = _imag_step(S, o_hist, a_prev, f, xi_n, g)
            Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731
        S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
        f_all = flat(Fs)
        r, d = _reward_term(S_all, f_all, flat(As), flat(Aps))
        r, d = r.view(HZ, B), d.view(HZ, B)
        alive = torch.cumprod(1 - d, 0)
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
        J = (disc * c_t * r).sum(0)
        c = alive[-1]
        r_sum = (c_t * r).detach().sum(0)
        rews, dones = r.detach(), d.detach()
        if not args.no_critic:
            J = J + (args.gamma**HZ) * c * critic(_critic_input(S, f, xi_n, g))
        loss = -J.mean()
        opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(pi.parameters(), 0.1)
        if torch.isfinite(gn):
            opt_pi.step()
        set_requires_grad(wm, True)
        set_requires_grad(critic, True)
        # 评论家：想象轨迹上的 GAE(λ) 目标
        v_loss = torch.zeros((), device=dev)
        if not args.no_critic:
            with torch.no_grad():
                S_c = tuple(
                    torch.cat((S0[i].unsqueeze(0), torch.stack([St[i].detach() for St in Ss])), 0).flatten(0, 1)
                    for i in range(6)
                )
                f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
                g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
                xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
                X = _critic_input(S_c, f_c, xi_c, g_c).view(HZ + 1, B, -1)
            critic.norm.update(X)
            with torch.no_grad():
                V = critic(X)
                R = V[-1]
                targets = []
                for t in range(HZ - 1, -1, -1):
                    R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
                    targets.append(R)
                targets = torch.stack(targets[::-1])
            Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
            n = Xf.shape[0]
            mb = max(n // args.critic_minibatches, 1)
            for _ in range(args.critic_epochs):
                perm = torch.randperm(n, device=dev)
                for i in range(args.critic_minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    if len(idx) == 0:
                        continue
                    v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                    opt_v.zero_grad(set_to_none=True)
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    opt_v.step()
        return loss.detach(), v_loss.detach(), (r_sum / HZ / MAX_STEP_REWARD).mean(), gn.detach(), c.mean().detach()

    # 采集
    buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev, mdp.state_dim, spec.na, mdp.xi_dim)
    obs_hist = torch.zeros(args.num_envs, Hh, mdp.obs_dim, device=dev)
    stats = {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}
    agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": []}

    @torch.no_grad()
    def collect(n_steps: int, st: dict, use_teacher: bool = False):
        nonlocal obs_hist, sim_steps
        for _ in range(n_steps):
            S = env.state()
            o = mdp.add_noise(mdp.obs(S, env.a_prev))
            fresh = env.ep_step == 0
            obs_hist = torch.where(
                fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1)
            )
            g = cmd.goal(env.f, K)
            pi.obs_norm.update(o)
            pi.goal_norm.update(g)
            a = pi(obs_hist, g)
            a = torch.clamp(a + args.action_noise * torch.randn_like(a), -ACTION_CLIP, ACTION_CLIP)
            if use_teacher and teacher is not None:
                if teacher.prev_action is not None:
                    teacher.prev_action[fresh] = 0.0  # 刚 reset 的 env 把老师的"上一步动作"清零
                a_t = teacher.act(S, cmd.cmd_at(env.f))
                sel = torch.rand(env.N, device=dev) < args.teacher_mix
                a = torch.where(sel[:, None], a_t, a)
                teacher.prev_action = a.clone()  # 老师的下一条 obs 要用"真正执行的动作"
            out = env.step(a)
            buffer.add(out)
            S0, S1 = mdp.unflatten_state(out["s"]), mdp.unflatten_state(out["s1"])
            wm.in_norm.update(wm.inputs(S0, out["a"], normalize_xi(out["xi"])))
            wm.out_norm.update(wm.targets(S0, S1))
            sim_steps += args.num_envs
            st["r"].append(out["r"].mean() / MAX_STEP_REWARD)
            if "ep_len" in out:
                st["ep_len"].append(out["ep_len"].mean())
                st["ep_ret"].append(out["ep_ret_norm"].mean())
                st["term"].append(out["term"].float().sum())
                st["done"].append(out["done"].float().sum())

    if start_it == 0:
        collect(args.warmup_steps, stats, use_teacher=teacher is not None)
    log_path = os.path.join(args.log_dir, "train_log.csv")
    log_f = open(log_path, "a")
    # 表头看文件是否为空，而不是看 start_it：resume 到新目录时 start_it != 0，
    # 旧逻辑会不写表头，导致曲线无法解析（踩过）
    if os.path.getsize(log_path) == 0:
        log_f.write(
            "iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,"
            "imag_survival,sim_rew,ep_len,ep_ret_norm,term_frac,it_per_s,lr,fps\n"
        )
    t_start = time.time() - wall_offset
    t_log = time.time()
    step_acc = 0.0
    for it in range(start_it, args.iterations):
        lr = lr_at(it)
        for o in (opt_wm, opt_pi, opt_v):
            for gp in o.param_groups:
                gp["lr"] = lr
        step_acc += 1.0 / args.utd
        n = int(step_acc)
        step_acc -= n
        collect(n, stats, use_teacher=teacher is not None and it < args.teacher_mix_iters)
        for _ in range(args.wm_batches):
            l, e8 = wm_update()
        agg["wm"].append(l), agg["wm8"].append(e8)
        if it >= args.wm_only_iters:
            pl, vl, ir, gn, surv = policy_update()
            agg["pi"].append(pl), agg["v"].append(vl), agg["ir"].append(ir)
            agg["gn"].append(gn), agg["surv"].append(surv)
        if (it + 1) % args.log_every == 0:
            mean = lambda x: float(sum(float(v) for v in x) / max(len(x), 1))  # noqa: E731
            dt_log = time.time() - t_log
            ips = args.log_every / dt_log
            t_log = time.time()
            wall_h = (time.time() - t_start) / 3600
            term_frac = float(sum(float(v) for v in stats["term"]) / max(sum(float(v) for v in stats["done"]), 1))
            row = [
                it + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]), mean(agg["v"]),
                mean(agg["ir"]), mean(agg["gn"]), mean(agg["surv"]), mean(stats["r"]), mean(stats["ep_len"]),
                mean(stats["ep_ret"]), term_frac, ips, lr, ips * args.num_envs * (1.0 / args.utd),
            ]  # fmt: skip
            log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
            log_f.flush()
            print(
                f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} "
                f"imag_r {row[7]:.3f} surv {row[9]:.3f} | sim_r {row[10]:.3f} ep_len {row[11]:.1f} "
                f"ep_ret {row[12]:.3f} term {term_frac:.2f} | {ips:.2f} it/s",
                flush=True,
            )
            stats = {k: [] for k in stats}
            agg = {k: [] for k in agg}
        _full = (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations
        # last.pt 存得比 ckpt_XXXXXX 勤：只掉 200 轮，而不是 1000 轮
        if _full or (it + 1) % args.last_every == 0:
            ck = {
                "wm": wm.state_dict(), "pi": pi.state_dict(), "critic": critic.state_dict(),
                "opt_wm": opt_wm.state_dict(), "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(),
                "iteration": it + 1, "wall_time": time.time() - t_start, "sim_steps": sim_steps,
                "args": vars(args), "spec": vars(spec), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }  # fmt: skip
            last = os.path.join(args.log_dir, "last.pt")
            torch.save(ck, last + ".tmp")
            os.replace(last + ".tmp", last)
            if _full:
                small = {k: ck[k] for k in ("pi", "wm", "iteration", "wall_time", "sim_steps",
                                            "args", "spec", "saved_at")}  # fmt: skip
                path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
                torch.save(small, path + ".tmp")
                os.replace(path + ".tmp", path)
    log_f.close()


if __name__ == "__main__":
    import threading

    from supermimic_wheelbipe import exit_fast

    code = 0
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        code = 1
    # Kit 的 close() 在跑过物理之后经常卡好几分钟；给 20 s 让它刷缓存，然后硬退出
    th = threading.Thread(target=simulation_app.close, daemon=True)
    th.start()
    th.join(20.0)
    exit_fast(code)
