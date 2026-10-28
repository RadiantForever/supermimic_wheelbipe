# SPDX-License-Identifier: MIT
"""在 MuJoCo 里对 SuperMimic 策略做 DAgger 微调（纯 CPU，几十分钟），用预训练 PPO 当示范。"""

from __future__ import annotations

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from supermimic_wheelbipe import networks  # noqa: E402
from supermimic_wheelbipe.command import CommandLib  # noqa: E402
from supermimic_wheelbipe.mdp import MDP  # noqa: E402
from supermimic_wheelbipe.mujoco_env import DEFAULT_XML, MujocoWheelbipe, load_policy_ckpt  # noqa: E402
from supermimic_wheelbipe.teacher import Teacher, pick_best_teacher  # noqa: E402

TERM_COS = float(np.cos(np.deg2rad(40.0)))
HIST_LEN = 10


# 指令
def make_cmd_line(T: int, rng: np.random.RandomState, vx_r, wz_r, h_r, seg=(150, 400)) -> np.ndarray:
    """分段常数指令（50 Hz）。段长 3~8 s，与训练时 resampling 5~15 s 同量级但更密。"""
    cmd = np.zeros((T, 4), dtype=np.float32)
    t = 0
    while t < T:
        n = int(rng.randint(seg[0], seg[1]))
        n = min(n, T - t)
        if rng.rand() < 0.15:
            v = [0.0, 0.0, 0.0, rng.uniform(*h_r)]
        else:
            v = [rng.uniform(*vx_r), 0.0, rng.uniform(*wz_r), rng.uniform(*h_r)]
        cmd[t:t + n] = v
        t += n
    return cmd


def goal_at(cmd: np.ndarray, f: int, K: int) -> np.ndarray:
    T = len(cmd)
    return np.concatenate([cmd[min(f + i, T - 1)] for i in range(1, K + 1)])


# 采集
def collect_episode(env, mdp, pi, teacher, spec, K, T, beta, cmd, rng, term_h):
    env.reset()
    if teacher is not None:
        teacher.reset()
    hist = None
    a_prev = torch.zeros(1, spec.na)
    obs_l, goal_l, act_l = [], [], []
    for f in range(T):
        S = env.state()
        o = mdp.obs(S, a_prev)
        hist = o.unsqueeze(1).repeat(1, HIST_LEN, 1) if hist is None else torch.cat(
            (hist[:, 1:], o.unsqueeze(1)), 1)
        g = goal_at(cmd, f, K)
        g_t = torch.tensor(g, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a_stu = pi(hist, g_t).clamp(-3.0, 3.0)
            a_tea = teacher.act(S, torch.tensor(cmd[min(f, len(cmd) - 1)]).unsqueeze(0)).clamp(-3.0, 3.0) \
                if teacher is not None else a_stu
        obs_l.append(o[0].numpy().copy())
        goal_l.append(g.copy())
        act_l.append(a_tea[0].numpy().copy())

        use_teacher = (teacher is not None) and (rng.rand() < beta)
        a_exec = a_tea if use_teacher else a_stu
        if not use_teacher and teacher is not None:
            # 老师观测里含"上一步动作"，执行了学生的动作就要让老师知道
            teacher.prev_action = a_stu.detach().clone()
        env.step(a_exec)
        a_prev = a_exec
        if env.base_z < term_h or env.tilt_cos < TERM_COS:
            break
    return (np.stack(obs_l) if obs_l else np.zeros((0, mdp.obs_dim), np.float32),
            np.stack(goal_l) if goal_l else np.zeros((0, 4 * K), np.float32),
            np.stack(act_l) if act_l else np.zeros((0, spec.na), np.float32))


# 训练
def build_hist(obs_pad, ep, t, device):
    """(B, 10, 78)：t-9..t，越界用 t=0 的帧复制（与推理时的 warm start 一致）。"""
    idx = np.clip(t[:, None] - np.arange(HIST_LEN - 1, -1, -1)[None, :], 0, None)
    return torch.tensor(obs_pad[ep[:, None], idx], dtype=torch.float32, device=device)


def train_student(pi, episodes, K, steps, batch, lr, device, log_every=100, loss_kind="l1"):
    norm_params = {id(p) for m in (pi.obs_norm, pi.goal_norm) for p in m.parameters()}
    params = [p for p in pi.parameters() if id(p) not in norm_params]
    opt = torch.optim.Adam(params, lr=lr)
    lengths = [len(e[0]) for e in episodes]
    Tmax = max(lengths)
    obs_pad = np.zeros((len(episodes), Tmax, episodes[0][0].shape[1]), dtype=np.float32)
    goal_pad = np.zeros((len(episodes), Tmax, 4 * K), dtype=np.float32)
    act_pad = np.zeros((len(episodes), Tmax, episodes[0][2].shape[1]), dtype=np.float32)
    for i, (o, g, a) in enumerate(episodes):
        obs_pad[i, :len(o)] = o
        obs_pad[i, len(o):] = o[-1]
        goal_pad[i, :len(g)] = g
        act_pad[i, :len(a)] = a
    starts = np.zeros(len(episodes), dtype=np.int64)
    starts[1:] = np.cumsum(lengths)[:-1]
    flat_len = int(sum(lengths))

    pi.train()
    for it in range(1, steps + 1):
        pick = np.random.randint(0, flat_len, batch)
        ep = np.searchsorted(starts, pick, side="right") - 1
        t = pick - starts[ep]
        hist = build_hist(obs_pad, ep, t, device)
        goal = torch.tensor(goal_pad[ep, t], dtype=torch.float32, device=device)
        target = torch.tensor(act_pad[ep, t], dtype=torch.float32, device=device)
        pred = pi(hist, goal)
        if loss_kind == "l1":
            # 老师是确定性策略，L1 不会像 MSE 那样把极端动作往均值收缩
            loss = torch.nn.functional.smooth_l1_loss(pred, target, beta=0.05)
        else:
            loss = ((pred - target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if it % log_every == 0 or it == 1:
            with torch.no_grad():
                e = (pred - target).abs().mean(0)
            legs = e[:4].mean().item()
            print(f"      step {it:5d}  mse={loss.item():.4f}  |Δa| 腿={legs:.3f} 轮={e[4:].mean().item():.3f}")
    pi.eval()
    return float(loss.item())


# 评测
EVAL_CMDS = [
    ("vx+1.4", 1.4, 0.0), ("vx+1.0", 1.0, 0.0), ("vx+0.5", 0.5, 0.0),
    ("vx-0.5", -0.5, 0.0), ("vx-1.0", -1.0, 0.0), ("vx 0 ", 0.0, 0.0),
    ("wz+2.0", 0.0, 2.0), ("wz-2.0", 0.0, -2.0),
]


@torch.no_grad()
def evaluate(pi, spec, env, mdp, K, h_cmd, steps=200, verbose=True):
    res = []
    for name, vx, wz in EVAL_CMDS:
        env.reset()
        hist, a_prev = None, torch.zeros(1, spec.na)
        vs, ws, ok = [], [], True
        for f in range(steps):
            S = env.state()
            o = mdp.obs(S, a_prev)
            hist = o.unsqueeze(1).repeat(1, HIST_LEN, 1) if hist is None else torch.cat(
                (hist[:, 1:], o.unsqueeze(1)), 1)
            g = torch.tensor([[vx, 0.0, wz, h_cmd]] * K, dtype=torch.float32).reshape(1, -1)
            a = pi(hist, g).clamp(-3.0, 3.0)
            env.step(a)
            a_prev = a
            vs.append(env.vel_forward)
            ws.append(float(env.d.qvel[3:6] @ env.d.xmat[env.base_id].reshape(3, 3)[2]))
            if env.base_z < 0.10 or env.tilt_cos < TERM_COS:
                ok = False
                break
        k = max(1, min(len(vs) // 3, 40))
        v = float(np.mean(vs[-k:])) if vs else 0.0
        w = float(np.mean(ws[-k:])) if ws else 0.0
        res.append((name, v - vx if vx else 0.0, w - wz if wz else 0.0, ok))
    if verbose:
        e_lin = np.mean([abs(r[1]) for r in res if r[0].startswith("vx")])
        e_ang = np.mean([abs(r[2]) for r in res if r[0].startswith("wz")])
        falls = sum(0 if r[3] else 1 for r in res)
        print(f"    [eval] |e_lin|={e_lin:.3f}  |e_ang|={e_ang:.3f}  摔倒 {falls}/{len(res)}   "
              + "  ".join(f"{r[0]}:{r[1]:+.2f}/{r[2]:+.2f}" for r in res))
    return res


# 主流程
def main():
    ap = argparse.ArgumentParser(description="MuJoCo 里 DAgger 微调 SuperMimic 策略")
    ap.add_argument("--ckpt", type=str,
                    default=os.path.join("..", "export", "wb_export", "policies", "wb_c3__last.pt"))
    ap.add_argument("--out_dir", type=str, default="logs/mj_dagger")
    ap.add_argument("--teacher", type=str, default="auto")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=12, help="每轮采多少条 episode")
    ap.add_argument("--ep_len", type=int, default=400, help="每条 episode 多少控制步（400=8s）")
    ap.add_argument("--train_steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--beta_start", type=float, default=1.0)
    ap.add_argument("--beta_end", type=float, default=0.3)
    ap.add_argument("--h_range", type=float, nargs=2, default=(0.16, 0.20),
                    help="采样高度区间。**必须限制在老师能干的区间**：实测 h=0.24 连老师都"
                         "跑不了 1.4 m/s（1.4 s 就摔），在那种地方采标签只会教坏学生")
    ap.add_argument("--vx_range", type=float, nargs=2, default=(-1.2, 1.4))
    ap.add_argument("--wz_range", type=float, nargs=2, default=(-2.5, 2.5))
    ap.add_argument("--eval_h", type=float, nargs="+", default=(0.18, 0.20))
    ap.add_argument("--eval_steps", type=int, default=300)
    ap.add_argument("--teacher_frac", type=float, default=0.25,
                    help="每轮里有多少比例的 episode 完全用老师跑（保证数据质量不会随 β 下降而崩）")
    ap.add_argument("--loss", type=str, default="l1", choices=["l1", "mse"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--xml", type=str, default=DEFAULT_XML)
    a = ap.parse_args()

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    networks.AMP_DTYPE = None
    rng = np.random.RandomState(a.seed)

    ck = a.ckpt if os.path.isabs(a.ckpt) else os.path.normpath(os.path.join(REPO, a.ckpt))
    out_dir = a.out_dir if os.path.isabs(a.out_dir) else os.path.normpath(os.path.join(REPO, a.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    ck0 = torch.load(ck, map_location=a.device)
    pi, spec, args = load_policy_ckpt(ck, a.device)
    K, Hh = int(args["goal_frames"]), int(args["hist"])
    assert Hh == HIST_LEN, f"这份代码假设 hist={HIST_LEN}，ckpt 是 {Hh}"

    # 冻结归一化层
    for m in (pi.obs_norm, pi.goal_norm):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    env = MujocoWheelbipe(spec, xml=a.xml)
    z_stand = float(env.rest_height())
    term_h = z_stand - 0.06
    dummy = CommandLib.fixed([[0.0, 0.0, 0.0, a.eval_h[0]]], device=a.device, frames=10)
    mdp = MDP(spec, dummy, kin=None, default_q=torch.zeros(spec.nj), qd_limit=None)
    teacher = Teacher(spec, spec.joint_names, torch.zeros(spec.nj),
                      pick_best_teacher(a.teacher), device=a.device, verbose=False)
    print(f"[setup] env={env.m.njnt} joints  obs={mdp.obs_dim} goal={4 * K} hist={Hh}  "
          f"z_stand={z_stand:.3f} term={term_h:.3f}")
    print(f"[setup] student={os.path.basename(ck)} it={args['iteration']}  "
          f"teacher={teacher.path}")

    print("\n[before] 原始策略：")
    for h in a.eval_h:
        print(f"  h={h:.2f}")
        evaluate(pi, spec, env, mdp, K, h, a.eval_steps)

    best = float("inf")
    for it in range(a.iters):
        beta = a.beta_start * (a.beta_end / a.beta_start) ** (it / max(1, a.iters - 1))
        t0 = time.perf_counter()
        episodes, total, n_tea = [], 0, 0
        for ep in range(a.episodes):
            T = a.ep_len
            cmd = make_cmd_line(T, rng, a.vx_range, a.wz_range, a.h_range)
            b_ep = 1.0 if rng.rand() < a.teacher_frac else beta
            n_tea += int(b_ep >= 1.0)
            o, g, act = collect_episode(env, mdp, pi, teacher, spec, K, T, b_ep, cmd, rng, term_h)
            if len(o) >= 50:
                episodes.append((o, g, act))
                total += len(o)
        t_col = time.perf_counter() - t0
        print(f"\n[iter {it + 1}/{a.iters}] beta={beta:.2f}（其中 {n_tea} 条纯老师）"
              f"  采集 {len(episodes)} 条 / {total} 帧 ({t_col:.0f}s)")

        t0 = time.perf_counter()
        train_student(pi, episodes, K, a.train_steps, a.batch, a.lr, a.device, loss_kind=a.loss)
        print(f"    训练 {a.train_steps} 步 ({time.perf_counter() - t0:.0f}s)")
        res = evaluate(pi, spec, env, mdp, K, a.eval_h[0], a.eval_steps)

        n_fall = sum(0 if r[3] else 1 for r in res)
        score = float(np.mean([abs(r[1]) for r in res if r[0].startswith("vx")])
                      + 0.2 * np.mean([abs(r[2]) for r in res if r[0].startswith("wz")])
                      + 0.3 * n_fall)
        ck_out = {"pi": pi.state_dict(), "spec": ck0["spec"], "args": ck0["args"],
                  "iteration": int(args["iteration"])}
        torch.save(ck_out, os.path.join(out_dir, "last.pt"))
        if score < best:
            best = score
            torch.save(ck_out, os.path.join(out_dir, "best.pt"))
        print(f"    -> {out_dir}/last.pt  (score={score:.3f}, best={best:.3f})")

    print("\n[after] last.pt：")
    for h in a.eval_h:
        print(f"  h={h:.2f}")
        evaluate(pi, spec, env, mdp, K, h, a.eval_steps)
    print(f"\n产物: {out_dir}/last.pt（最后一轮）, {out_dir}/best.pt（评测最优）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
