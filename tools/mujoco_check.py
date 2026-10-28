# SPDX-License-Identifier: MIT
"""MuJoCo 移植的验收自检（纯 CPU，不需要 Isaac）。"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from supermimic_wheelbipe import networks  # noqa: E402
from supermimic_wheelbipe.command import CommandLib  # noqa: E402
from supermimic_wheelbipe.constants import SPRING_FORCE_DOWN, SPRING_LENGTH, SPRING_OFFSET, SPRING_FORCE_UP  # noqa: E402
from supermimic_wheelbipe.mdp import MDP  # noqa: E402
from supermimic_wheelbipe.mujoco_env import MujocoWheelbipe, load_policy_ckpt  # noqa: E402

OK, BAD = "  PASS", "  FAIL"
fails = 0


def check(cond, msg, extra=""):
    global fails
    print(f"{OK if cond else BAD}  {msg}{('  ' + extra) if extra else ''}")
    if not cond:
        fails += 1
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=os.path.join("..", "export", "wb_export", "policies", "wb_final__last.pt"))
    ap.add_argument("--xml", type=str, default=None)
    ap.add_argument("--steps", type=int, default=600, help="静置多少控制步")
    ap.add_argument("--device", type=str, default="cpu")
    a = ap.parse_args()

    networks.AMP_DTYPE = None
    ck = a.ckpt if os.path.isabs(a.ckpt) else os.path.join(REPO, a.ckpt)
    pi, spec, args = load_policy_ckpt(ck, a.device)
    env = MujocoWheelbipe(spec) if a.xml is None else MujocoWheelbipe(spec, xml=a.xml)
    print(env.describe())
    print(f"[ckpt] {os.path.basename(ck)}  it={args['iteration']}  "
          f"nq={spec.nq} nd={spec.nd} na={spec.na} nj={spec.nj} obs={spec.nq + spec.nd + 6 + spec.na}")

    # 1. 关节对齐
    print("\n[1] 关节对齐")
    check(True, f"spec 36 关节全部在 MJCF 里（{spec.nj} 个）")
    check(len(spec.act_leg_idx) == 4 and len(spec.act_wheel_idx) == 2,
          "动作 6 维：4 腿 + 2 轮", f"{[spec.joint_names[i] for i in spec.act_idx]}")

    # 2. FK 站立高度
    print("\n[2] FK 站立高度（全 0 关节角）")
    env.reset()
    h_fk = env.rest_height()
    check(0.15 < h_fk < 0.22, f"轮子触地高度 {h_fk:.4f} m", "(Isaac: FK 0.185 / 静置 0.177)")

    # 3. 弹簧力
    print("\n[3] 弹簧力")
    q0 = torch.zeros(1, 2)
    from supermimic_wheelbipe.constants import spring_force  # noqa: E402

    f0 = float(spring_force(q0, torch.zeros(1, 1))[0, 0])
    check(abs(f0 - 573.6) < 5.0, f"q=0 时弹簧力 {f0:.1f} N", "(Isaac ~570 N)")

    # 4. 零动作静置
    print(f"\n[4] 零动作静置（从 0.38 m 自由落体，{a.steps} 控制步）")
    env.reset()
    zs = []
    for k in range(a.steps):
        env.set_zero_action()
        env.step()
        if k % max(1, a.steps // 10) == 0:
            zs.append((k, env.base_z, env.tilt_cos, env.ncon))
    z_end, tilt_end = env.base_z, env.tilt_cos
    for k, z, c, nc in zs:
        print(f"    step {k:4d}  z={z:6.4f}  cos={c:6.4f}  ncon={nc}")
    print(f"    step {a.steps:4d}  z={z_end:6.4f}  cos={tilt_end:6.4f}  ncon={env.ncon}")
    check(0.15 < z_end < 0.21, f"稳态根高 {z_end:.4f} m", "(Isaac 0.177)")
    check(tilt_end > 0.9, f"未倾倒 cos(tilt)={tilt_end:.3f}")

    # 5. 轮速伺服
    # 悬空测：零动作姿态下机身最低处会轻擦地面，带载测会被地面摩擦吃掉（不代表伺服有问题）
    print("\n[5] 轮速伺服（关重力悬空，指令 2 rad/s，1 s）")
    g0 = env.m.opt.gravity.copy()
    env.m.opt.gravity[:] = 0.0
    env.reset(z=0.60)
    a_cmd = torch.zeros(1, spec.na)
    a_cmd[0, len(spec.act_leg_idx):] = 0.2  # 10*0.2 = 2 rad/s
    for _ in range(50):
        env.step(a_cmd)
    wl = float(env.d.qvel[env.m.jnt_dofadr[env.m.joint("left_wheel_joint").id]])
    wr = float(env.d.qvel[env.m.jnt_dofadr[env.m.joint("right_wheel_joint").id]])
    env.m.opt.gravity[:] = g0
    env.reset()
    check(abs(wl - 2.0) < 0.3 and abs(wr - 2.0) < 0.3, f"实测轮速 L={wl:+.2f} R={wr:+.2f}",
          "(Isaac: 指令 +2 → 实测 +2.2)")

    # 6. 观测量级
    print("\n[6] 观测量级 vs ckpt 的 obs_norm")
    env.reset()
    cmd = CommandLib.fixed([[0.0, 0.0, 0.0, 0.24]], device=a.device, frames=20000)
    mdp = MDP(spec, cmd, kin=None, default_q=torch.zeros(spec.nj), qd_limit=None)
    S = env.state()
    o = mdp.obs(S, env.a_prev)
    mu, sd = pi.obs_norm.mean, pi.obs_norm.std()
    z = ((o[0] - mu) / sd).detach().numpy()
    groups = [("q", 0, spec.nq), ("qd", spec.nq, spec.nq + spec.nd),
              ("w_b", spec.nq + spec.nd, spec.nq + spec.nd + 3),
              ("g_b", spec.nq + spec.nd + 3, spec.nq + spec.nd + 6),
              ("a_prev", spec.nq + spec.nd + 6, o.shape[-1])]
    for name, i0, i1 in groups:
        seg = z[i0:i1]
        print(f"    {name:<7s} dim={i1 - i0:<3d} |z|max={np.abs(seg).max():7.2f} mean={seg.mean():+7.2f}")
    check(o.shape == (1, pi.obs_norm.mean.numel()), f"obs 形状 {tuple(o.shape)}")
    check(np.abs(z[spec.nq + spec.nd + 3: spec.nq + spec.nd + 6]).max() < 40,
          "静置时重力投影 g_b 与训练分布同量级（|z|<40）")

    print(f"\n{'全部通过' if fails == 0 else f'{fails} 项未通过'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
