# SPDX-License-Identifier: MIT
"""老师策略（已训好的 PPO）在本框架里的三种用法。"""

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

if "--list" in sys.argv:  # 不需要 Isaac，先处理掉
    from supermimic_wheelbipe.teacher import find_pretrained

    runs = find_pretrained()
    print(f"找到 {len(runs)} 个候选老师（前 20）：")
    base = os.path.normpath(os.path.join(REPO, "..", "wheeled-legged_RL", "pretrained"))
    for r in runs[:20]:
        jit = "JIT  " if os.path.exists(os.path.join(r, "exported", "policy.pt")) else "ckpt "
        print(f"  [{jit}] {os.path.relpath(r, base)}")
    sys.exit(0)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["play", "record", "bc"], default="play")
parser.add_argument("--teacher", type=str, default="auto", help="run 目录 / .pt 文件 / auto")
parser.add_argument("--list", action="store_true", help="只列出候选老师并退出")
parser.add_argument("--cmd", type=float, nargs=3, default=(1.0, 0.0, -1.0), metavar=("VX", "WZ", "H"))
parser.add_argument("--grid", action="store_true",
                    help="play 模式：跑 16 条指令网格（7 直行 + 6 原地转 + 2 组合 + 站立），打印判定表")
parser.add_argument("--height_cmd", type=float, default=None,
                    help="指令高度，默认 z_stand-0.04（不能叫 --height，会和 SimulationApp 冲突）")
parser.add_argument("--seconds", type=float, default=30.0)
parser.add_argument("--out", type=str, default="")
# 环境
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_spacing", type=float, default=2.0)
parser.add_argument("--friction", type=float, default=1.0)
parser.add_argument("--z_stand", type=float, default=None)
parser.add_argument("--settle_steps", type=int, default=120)
parser.add_argument("--term_height", type=float, default=None)
parser.add_argument("--com_random", type=float, nargs=3, default=(0.02, 0.03, 0.03))
parser.add_argument("--spring_bias", type=float, default=50.0)
parser.add_argument("--max_episode_steps", type=int, default=500)
parser.add_argument("--no_gimbal_usd", type=int, default=0)
parser.add_argument("--no_kin", action="store_true")
parser.add_argument("--model_json", type=str, default=os.path.join(REPO, "data", "wheelbipe_v14_2_model.json"))
# 指令分布（bc 用）
parser.add_argument("--num_clips", type=int, default=128)
parser.add_argument("--clip_frames", type=int, default=500)
parser.add_argument("--lin_vel_x", type=float, nargs=2, default=(-2.7, 2.7))
parser.add_argument("--ang_vel_z", type=float, nargs=2, default=(-6.2832, 6.2832))
parser.add_argument("--seed", type=int, default=0)
# 策略超参（bc 保存，train 时要用一致的值）
parser.add_argument("--goal_frames", type=int, default=10)
parser.add_argument("--hist", type=int, default=10)
# bc
parser.add_argument("--bc_iters", type=int, default=150)
parser.add_argument("--bc_steps", type=int, default=4, help="每次迭代采集的仿真步数")
parser.add_argument("--bc_keep", type=int, default=32, help="缓冲保留的步数")
parser.add_argument("--bc_grad_steps", type=int, default=8)
parser.add_argument("--bc_batch", type=int, default=2048)
parser.add_argument("--lr", type=float, default=1e-3)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

import supermimic_wheelbipe.networks as networks  # noqa: E402
from supermimic_wheelbipe.command import CommandLib, eval_pairs  # noqa: E402
from supermimic_wheelbipe.constants import default_height_cmd, default_term_height  # noqa: E402
from supermimic_wheelbipe.env import TrackingSim, make_sim_cfg, make_world, resolve_stand_height  # noqa: E402
from supermimic_wheelbipe.kinematics import Kinematics  # noqa: E402
from supermimic_wheelbipe.mdp import MAX_STEP_REWARD, MDP, RewardCfg  # noqa: E402
from supermimic_wheelbipe.model import ensure_model, wheel_bodies  # noqa: E402
from supermimic_wheelbipe.networks import Policy  # noqa: E402
from supermimic_wheelbipe.robot import CONTROL_DT, build_spec, make_robot_cfg, usd_path  # noqa: E402
from supermimic_wheelbipe.teacher import Teacher, pick_best_teacher  # noqa: E402


def pick_teacher(arg: str) -> str:
    try:
        return pick_best_teacher(arg)
    except FileNotFoundError as e:
        raise SystemExit(f"{e}（--teacher 指向 pretrained/<task>/<run> 目录）") from e


def build_all(dev, num_envs, eval_mode: bool, fixed_cmd=None, grid: bool = False):
    usd = usd_path(no_gimbal=bool(args.no_gimbal_usd))
    sim, scene, robot = make_world(make_sim_cfg(dev, friction=args.friction), num_envs, make_robot_cfg(usd),
                                   env_spacing=args.env_spacing)  # fmt: skip
    lim = robot.data.joint_pos_limits
    lim = lim.torch if hasattr(lim, "torch") else lim
    spec = build_spec(robot.joint_names, q_lo=lim[0, :, 0].clone(), q_hi=lim[0, :, 1].clone())
    default_q = robot.data.default_joint_pos
    default_q = (default_q.torch if hasattr(default_q, "torch") else default_q)[0].clone()
    kin = None
    if not args.no_kin:
        try:
            model = ensure_model(usd, path=args.model_json)
            kin = Kinematics(model, list(robot.joint_names), dev, tracked=list(robot.body_names),
                             wheel_bodies=wheel_bodies(model, spec))
        except Exception as e:  # noqa: BLE001
            print(f"[teacher] FK 不可用（{e}）")
    z_stand, _ = resolve_stand_height(sim, scene, robot, spec, kin, default_q,
                                      settle_steps=args.settle_steps, override=args.z_stand)
    term_height = args.term_height if args.term_height is not None else default_term_height(z_stand)
    frames = int(args.seconds / CONTROL_DT) + 64
    if grid:
        h = args.height_cmd if args.height_cmd is not None else default_height_cmd(z_stand)
        cmd = CommandLib.fixed([[vx, 0.0, wz, h] for vx, wz in eval_pairs()], dev, frames=frames)
        print(f"[grid] {len(cmd.names)} 条指令, height={h:.3f} m")
    elif fixed_cmd is None:
        cmd = CommandLib.random(dev, num_clips=args.num_clips, clip_frames=args.clip_frames, seed=args.seed,
                                lin_vel_x=tuple(args.lin_vel_x), ang_vel_z=tuple(args.ang_vel_z),
                                height=(z_stand - 0.08, z_stand - 0.02))  # fmt: skip
    else:
        vx, wz, h = fixed_cmd
        h = h if h > 0 else default_height_cmd(z_stand)
        cmd = CommandLib.fixed([[vx, 0.0, wz, h]], dev, frames=frames)
    mdp = MDP(spec, cmd, kin=kin, default_q=default_q, cfg=RewardCfg(term_height=term_height))
    env = TrackingSim(sim, scene, robot, spec, cmd, mdp, eval_mode=eval_mode,
                      max_episode_steps=args.max_episode_steps if not eval_mode else 10**9,
                      z_stand=z_stand, friction=args.friction, com_random=tuple(args.com_random),
                      spring_bias=args.spring_bias)  # fmt: skip
    return sim, scene, robot, spec, kin, mdp, env, cmd, z_stand, default_q


def main():
    dev = args.device
    teacher_dir = pick_teacher(args.teacher)
    eval_mode = args.mode in ("play", "record")
    fixed = tuple(args.cmd) if eval_mode and not args.grid else None
    n_envs = len(eval_pairs()) if (args.grid and eval_mode) else args.num_envs
    sim, scene, robot, spec, kin, mdp, env, cmd, z_stand, default_q = build_all(
        dev, n_envs, eval_mode, fixed, grid=args.grid and eval_mode
    )
    tea = Teacher(spec, list(robot.joint_names), default_q, teacher_dir, dev)

    if args.mode == "play":
        play(sim, env, cmd, tea, grid=args.grid)
    elif args.mode == "record":
        record(env, cmd, tea)
    else:
        bc(env, cmd, tea, mdp, spec, dev)


# play
@torch.no_grad()
def play(sim, env, cmd, tea, grid: bool = False):
    dev = env.device
    ids = torch.arange(env.N, device=dev)
    env.reset(ids, cmd.clip_start.clone())
    tea.reset()
    steps = int(args.seconds / CONTROL_DT)
    if grid:
        return play_grid(env, cmd, tea, steps)

    sim.set_camera_view(eye=[3.0, 3.0, 1.5], target=[0.0, 0.0, 0.3])
    ret, n, done_at = 0.0, 0, None
    for t in range(steps):
        S = env.state()
        a = tea.act(S, cmd.cmd_at(env.f))
        out = env.step(a)
        ret += float(out["r"][0])
        n += 1
        if t % 25 == 0:
            tm = out["terms"]
            print(f"  t={t * CONTROL_DT:5.2f}s  z={float(S[0][0, 2]):.3f}  vx_err={float(tm['e_lin'][0]):+.2f}  "
                  f"wz_err={float(tm['e_ang'][0]):+.2f}  h_err={float(tm['e_height'][0]):+.2f}", flush=True)
        p = S[0][0].tolist()
        sim.set_camera_view(eye=[p[0] + 2.5, p[1] + 2.5, 1.2], target=[p[0], p[1], 0.25])
        if bool(env.done_eval.any()):
            done_at = t
            break
    print(f"[play] 存活 {n * CONTROL_DT:.2f}s / {args.seconds:.0f}s, "
          f"平均步回报 {ret / max(n, 1) / MAX_STEP_REWARD:.3f}（1.0 = 上限）")
    if done_at is not None:
        print(f"[play] 第 {done_at} 步终止：老师在我们的环境里没能撑住。"
              "先跑 tools/calibrate.py，重点看弹簧力、z_stand、term_height。")
    else:
        print("[play] 全程没有触发终止 —— 说明 USD / 关节分组 / 弹簧力 / 动作语义都对齐了")


@torch.no_grad()
def play_grid(env, cmd, tea, steps):
    """16 条指令各跑一遍：老师能全过 = 环境移植正确（同时得到评测基线）。"""
    N = env.N
    alive = torch.ones(N, dtype=torch.bool, device=env.device)
    surv = torch.zeros(N, device=env.device)
    e_lin = torch.zeros(N, device=env.device)
    e_ang = torch.zeros(N, device=env.device)
    e_h = torch.zeros(N, device=env.device)
    cnt = torch.zeros(N, device=env.device)
    for _t in range(steps):
        S = env.state()
        a = tea.act(S, cmd.cmd_at(env.f))
        m = alive.clone().float()
        out = env.step(a)
        tm = out["terms"]
        if (_t + 1) % 100 == 0:
            print(f"  [grid] t={_t + 1}/{steps} " f"alive={int(alive.sum())}/{N}", flush=True)
        e_lin += torch.where(m > 0, tm["e_lin"].abs(), torch.zeros_like(e_lin))
        e_ang += torch.where(m > 0, tm["e_ang"].abs(), torch.zeros_like(e_ang))
        e_h += torch.where(m > 0, tm["e_height"].abs(), torch.zeros_like(e_h))
        cnt += m
        surv += m * CONTROL_DT
        alive &= ~out["done"]
        if not bool(alive.any()):
            break
    cnt = cnt.clamp(min=1)
    total = steps * CONTROL_DT
    print(f"\n{'指令':<26s} {'存活':>7s} {'|e_lin|':>9s} {'|e_ang|':>9s} {'|e_h|':>8s}")
    for i in range(N):
        print(f"{cmd.names[i]:<26s} {float(surv[i]):6.1f}s {float(e_lin[i] / cnt[i]):9.3f} "
              f"{float(e_ang[i] / cnt[i]):9.3f} {float(e_h[i] / cnt[i]):8.3f}")
    ok = int((surv >= 0.9 * total).sum())
    m_lin = float((e_lin / cnt).mean())
    print(f"\n[grid] 存活 ≥90% 的指令: {ok}/{N}；平均 |e_lin| {m_lin:.3f} m/s")
    if ok >= N - 1 and m_lin < 0.25:
        print("[grid] PASS —— 老师在这套环境里跑得对，说明 USD/关节分组/弹簧力/动作语义和原项目一致。"
              "可以继续 record / bc。")
    elif ok >= N * 0.6:
        print("[grid] PARTIAL —— 大部分指令能跑，个别工况（看上面 存活 短的行）有偏差。"
              "先看这些工况是不是 z_stand/term_height 边界，再决定是否继续。")
    else:
        print("[grid] FAIL —— 老师大面积摔。不要往下做 record/bc，先按这个顺序查："
              "① tools/calibrate.py 的 FK 误差与静置过程；② spring force @ q=0 是否为 ~570 N；"
              "③ --friction 是否 1.0；④ USD 是否用了 _ng 版（--no_gimbal_usd 0 试试）。")


# record
@torch.no_grad()
def record(env, cmd, tea):
    dev = env.device
    steps = int(args.seconds / CONTROL_DT)
    ids = torch.arange(env.N, device=dev)
    env.reset(ids, cmd.clip_start.clone())
    tea.reset()
    rec = {k: [] for k in ("p", "R", "v", "w", "q", "qd", "cmd", "a")}
    for t in range(steps):
        S = env.state()
        c = cmd.cmd_at(env.f)
        a = tea.act(S, c)
        out = env.step(a)
        for k, v in zip(("p", "R", "v", "w", "q", "qd"), S):
            rec[k].append(v[0].clone())
        rec["cmd"].append(c[0].clone())
        rec["a"].append(a[0].clone())
        if bool(env.done_eval.any()):
            print(f"[record] 第 {t} 步终止，提前结束（已录 {t * CONTROL_DT:.2f}s）")
            break
    out_path = args.out or os.path.join(REPO, "data", "teacher_ref.npz")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    import numpy as np

    data = {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}
    data["dt"] = CONTROL_DT
    np.savez_compressed(out_path, **data)
    print(f"[record] 写出 {out_path}：{data['p'].shape[0]} 帧 @ {1 / CONTROL_DT:.0f} Hz，"
          f"键={list(data)}")
    print("[record] 这是 SuperMimic 原版意义上的『参考动作』：轨迹跟踪任务可以把 command.py "
          "换成读它的 MotionLib，奖励换成参考关节角/根位姿误差。")


# bc
def bc(env, cmd, tea, mdp, spec, dev):
    K, Hh = args.goal_frames, args.hist
    GOAL_DIM = 4 * K
    networks.AMP_DTYPE = None
    pi = Policy(mdp.obs_dim, spec.na, GOAL_DIM, hist=Hh).to(dev)
    opt = torch.optim.Adam(pi.parameters(), lr=args.lr)
    D = Hh * mdp.obs_dim
    buf = {"h": [], "g": [], "y": []}
    env.reset(torch.arange(env.N, device=dev))
    tea.reset()
    hist = None
    print(f"[bc] {args.bc_iters} it x {args.bc_steps} 步 x {env.N} envs, batch {args.bc_batch}, lr {args.lr}")
    t0 = time.time()
    for it in range(args.bc_iters):
        with torch.no_grad():
            for _ in range(args.bc_steps):
                S = env.state()
                o = mdp.obs(S, env.a_prev)
                fresh = env.ep_step == 0
                hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.where(
                    fresh[:, None, None], o.unsqueeze(1).expand_as(hist), torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
                )
                g = cmd.goal(env.f, K)
                y = tea.act(S, cmd.cmd_at(env.f))  # 老师动作 + 内部更新 prev_action
                pi.obs_norm.update(o)
                pi.goal_norm.update(g)
                env.step(y)
                tea.prev_action[env.ep_step == 0] = 0.0  # 刚 reset 的 env，老师的"上一步动作"清零
                buf["h"].append(hist.reshape(env.N, D).clone())
                buf["g"].append(g.clone())
                buf["y"].append(y.clone())
            while len(buf["h"]) > args.bc_keep:
                for k in buf:
                    buf[k].pop(0)
        H = torch.stack(buf["h"])
        G = torch.stack(buf["g"])
        Y = torch.stack(buf["y"])
        T, N = H.shape[:2]
        losses = []
        for _ in range(args.bc_grad_steps):
            ti = torch.randint(0, T, (args.bc_batch,), device=dev)
            ei = torch.randint(0, N, (args.bc_batch,), device=dev)
            hb = H[ti, ei].reshape(-1, Hh, mdp.obs_dim)
            gb, yb = G[ti, ei], Y[ti, ei]
            pl = ((pi(hb, gb) - yb) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            pl.backward()
            torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
            opt.step()
            losses.append(float(pl))
        if (it + 1) % 10 == 0:
            print(f"  it {it + 1:4d} | mse {sum(losses) / len(losses):.5f} | "
                  f"{time.time() - t0:.1f}s", flush=True)
    out_path = args.out or os.path.join(REPO, "logs", "bc", "bc_init.pt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"pi": pi.state_dict(), "args": {"obs_dim": mdp.obs_dim, "na": spec.na, "goal_frames": K,
                                                "hist": Hh, "teacher": args.teacher, "bc_iters": args.bc_iters}},
               out_path)  # fmt: skip
    print(f"[bc] 写出 {out_path}")
    print("[bc] 下一步: python supermimic_wheelbipe/train.py --init_policy " + out_path)


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
