# SPDX-License-Identifier: MIT
"""CPU 自检：不启动 Isaac，只用 torch 跑一遍「FK + 指令库 + MDP + 世界模型/策略/评论家 + 缓冲」。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from supermimic_wheelbipe.buffer import ReplayBuffer
from supermimic_wheelbipe.command import CommandLib
from supermimic_wheelbipe.constants import CONTROL_DT, spring_force
from supermimic_wheelbipe.kinematics import Kinematics
from supermimic_wheelbipe.mdp import MAX_STEP_REWARD, MDP, RewardCfg, normalize_xi
from supermimic_wheelbipe.networks import Critic, Policy, WorldModel
from supermimic_wheelbipe.rotations import mat_to_quat, so3_log, quat_to_mat
from supermimic_wheelbipe.spec import build_spec
from supermimic_wheelbipe.teacher import find_pretrained

DEV = "cpu"

# 仿 wheelbipe 的关节顺序（左右腿 + 轮 + 弹簧 + 被动杆），并造一棵同构的运动学树
JOINTS = [
    "left_front1_joint", "right_front1_joint", "left_rear1_joint", "right_rear1_joint",
    "left_front2_joint", "right_front2_joint", "left_front3_joint", "right_front3_joint",
    "left_front4_joint", "right_front4_joint", "left_rear2_joint", "right_rear2_joint",
    "left_spring1_joint", "right_spring1_joint", "left_spring2_joint", "right_spring2_joint",
    "left_wheel_joint", "right_wheel_joint",
]  # fmt: skip
BODIES = [
    "base_link",
    "left_front1_link", "right_front1_link", "left_rear1_link", "right_rear1_link",
    "left_front2_link", "right_front2_link", "left_front3_link", "right_front3_link",
    "left_front4_link", "right_front4_link", "left_rear2_link", "right_rear2_link",
    "left_spring1_link", "right_spring1_link", "left_spring2_link", "right_spring2_link",
    "left_wheel_link", "right_wheel_link",
]  # fmt: skip


def fake_model():
    """造一棵树：base -> front1 -> front2 -> front3 -> front4 -> wheel；base -> spring1 -> spring2。"""
    bodies = [dict(name="base_link", parent=-1, joint=None, kind="root", axis=[0.0, 0.0, 0.0],
                   p_J=[0.0, 0.0, 0.0], q_J=[1.0, 0.0, 0.0, 0.0], p_B=[0.0, 0.0, 0.0],
                   q_B=[1.0, 0.0, 0.0, 0.0], lower=None, upper=None)]  # fmt: skip
    idx = {"base_link": 0}

    def add(name, parent, joint, kind, axis, p):
        idx[name] = len(bodies)
        bodies.append(dict(name=name, parent=idx[parent], joint=joint, kind=kind, axis=axis,
                           p_J=p, q_J=[1.0, 0.0, 0.0, 0.0], p_B=[0.0, 0.0, 0.0],
                           q_B=[1.0, 0.0, 0.0, 0.0], lower=-3.14 if kind == "revolute" else -0.08,
                           upper=3.14 if kind == "revolute" else 0.08))  # fmt: skip

    for side, sy in (("left", 0.15), ("right", -0.15)):
        add(f"{side}_front1_link", "base_link", f"{side}_front1_joint", "revolute", [0.0, 1.0, 0.0], [0.0, sy, 0.0])
        add(f"{side}_rear1_link", f"{side}_front1_link", f"{side}_rear1_joint", "revolute", [0.0, 1.0, 0.0], [0.11, 0.0, 0.0])
        add(f"{side}_front2_link", f"{side}_rear1_link", f"{side}_front2_joint", "revolute", [0.0, 1.0, 0.0], [0.11, 0.0, 0.0])
        add(f"{side}_front3_link", f"{side}_front2_link", f"{side}_front3_joint", "revolute", [0.0, 1.0, 0.0], [0.11, 0.0, 0.0])
        add(f"{side}_front4_link", f"{side}_front3_link", f"{side}_front4_joint", "revolute", [0.0, 1.0, 0.0], [0.11, 0.0, 0.0])
        add(f"{side}_rear2_link", f"{side}_front4_link", f"{side}_rear2_joint", "revolute", [0.0, 1.0, 0.0], [0.05, 0.0, 0.0])
        add(f"{side}_wheel_link", f"{side}_rear2_link", f"{side}_wheel_joint", "revolute", [0.0, 1.0, 0.0], [0.05, 0.0, 0.0])
        add(f"{side}_spring1_link", "base_link", f"{side}_spring1_joint", "prismatic", [0.0, 0.0, 1.0], [0.05, sy, 0.0])
        add(f"{side}_spring2_link", f"{side}_spring1_link", f"{side}_spring2_joint", "prismatic", [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    return dict(root="base_link", bodies=bodies)


def main():
    torch.manual_seed(0)
    nj = len(JOINTS)
    spec = build_spec(JOINTS, q_lo=torch.full((nj,), -3.14), q_hi=torch.full((nj,), 3.14))
    print(spec.describe())
    assert spec.na == 6, spec.na
    assert spec.nq == nj - 2, (spec.nq, nj)  # 去掉 2 个轮子
    assert spec.nd == nj

    kin = Kinematics(fake_model(), JOINTS, DEV, tracked=BODIES, wheel_bodies=["left_wheel_link", "right_wheel_link"])
    kin.forward(torch.zeros(4, 3), torch.eye(3).expand(4, 3, 3), torch.zeros(4, 3), torch.zeros(4, 3),
                torch.zeros(4, nj), torch.zeros(4, nj))  # fmt: skip
    z = kin.rest_height(torch.zeros(nj), 0.06, kin.wheel_names)
    print(f"rest_height = {z:.3f} m")
    assert 0.0 < z < 1.5

    cmd = CommandLib.random(DEV, num_clips=4, clip_frames=300, seed=0)
    mdp = MDP(spec, cmd, kin=kin, default_q=torch.zeros(nj), cfg=RewardCfg(term_height=0.15))
    print(f"na={spec.na} obs={mdp.obs_dim} state={mdp.state_dim} inv={mdp.inv_dim} wm_out={mdp.wm_out_dim} "
          f"critic_in={mdp.inv_dim + mdp.critic_extra}")

    B, K, Hh, HZ = 8, 10, 10, 6
    f = torch.randint(0, cmd.num_frames, (B,))
    goal = cmd.goal(f, K)
    a_prev = torch.zeros(B, spec.na)

    # 状态/观测/奖励
    p = torch.zeros(B, 3)
    p[:, 2] = 0.30 + 0.02 * torch.randn(B)
    R = torch.eye(3).expand(B, 3, 3) @ quat_to_mat(torch.cat((torch.ones(B, 1), 0.02 * torch.randn(B, 3)), -1))
    S = (p, R, 0.1 * torch.randn(B, 3), 0.1 * torch.randn(B, 3),
         0.05 * torch.randn(B, spec.nq), 0.5 * torch.randn(B, spec.nd))  # fmt: skip
    x = mdp.flatten_state(S)
    assert x.shape == (B, mdp.state_dim), x.shape
    S2 = mdp.unflatten_state(x)
    for i in range(6):
        assert torch.allclose(S[i], S2[i], atol=1e-5), i
    o = mdp.obs(S, a_prev)
    assert o.shape == (B, mdp.obs_dim)
    assert torch.isfinite(mdp.add_noise(o)).all()
    r, terms = mdp.reward(S, f, torch.zeros(B, spec.na), a_prev, terms=True)
    assert r.shape == (B,) and torch.isfinite(r).all(), r
    print(f"reward per step: {float(r.mean()):+.4f} (max {MAX_STEP_REWARD:.3f})")
    assert float(r.abs().max()) < 1.0
    assert mdp.termination(S).dtype == torch.bool
    xi_n = normalize_xi(torch.zeros(B, 5))
    cf = mdp.critic_features(S, f, xi_n)
    assert cf.shape == (B, mdp.inv_dim + mdp.critic_extra), cf.shape

    # 世界模型一步
    wm = WorldModel(mdp.inv_dim, spec.na, mdp.xi_dim, mdp.wm_out_dim, hidden=(64, 64))
    S1 = wm.step(S, torch.zeros(B, spec.na), xi_n)
    assert all(t.shape == s.shape for t, s in zip(S1, S)), [t.shape for t in S1]
    tgt = wm.targets(S, S1)
    assert tgt.shape == (B, mdp.wm_out_dim)
    out = wm.state_error(S1, S)
    assert out.shape == tgt.shape
    mdp.sanitize(S1)

    # 策略 / 评论家
    pi = Policy(mdp.obs_dim, spec.na, 4 * K, hist=Hh)
    hist = o.unsqueeze(1).repeat(1, Hh, 1)
    a = pi(hist, goal)
    assert a.shape == (B, spec.na) and torch.isfinite(a).all()
    critic = Critic(mdp.inv_dim + mdp.critic_extra + 4 * K)
    v = critic(torch.cat((cf, goal), -1))
    assert v.shape == (B,)

    # 缓冲
    buf = ReplayBuffer(64, B, DEV, mdp.state_dim, spec.na, mdp.xi_dim)
    for i in range(40):
        buf.add(dict(s=x, s1=mdp.flatten_state(S1), a=a, a_prev=a_prev, xi=torch.zeros(B, 5),
                     r=r, f=f, ep_step=torch.zeros(B, dtype=torch.int32),
                     done=torch.zeros(B, dtype=torch.bool)))  # fmt: skip
    w = buf.sample_windows(16, 8)
    assert w["s"].shape == (16, 8, mdp.state_dim) and w["mask"].shape == (16, 8)
    st, hh = buf.sample_starts(16, Hh)
    assert st["s"].shape == (16, mdp.state_dim) and hh["s"].shape == (16, Hh, mdp.state_dim)

    # 弹簧
    f0 = spring_force(torch.zeros(1, 2), torch.zeros(1, 1))
    print(f"spring force @ q=0: {f0.tolist()}")
    assert float(f0.min()) > 300.0

    # 旋转工具
    qq = torch.tensor([[0.9, 0.1, -0.3, 0.2]])
    Rr = quat_to_mat(qq)
    assert torch.allclose(mat_to_quat(Rr), qq / qq.norm(), atol=1e-5)
    assert so3_log(torch.eye(3).unsqueeze(0)).abs().max() < 1e-4

    # 老师策略（有预训练权重时才测）
    try:
        runs = find_pretrained()
    except Exception:  # noqa: BLE001
        runs = []
    if runs:
        from supermimic_wheelbipe.teacher import Teacher

        pick = [r for r in runs if "flat_and_rotation" in r] or runs
        print(f"\n[teacher] 候选 {len(runs)} 个，测 {os.path.basename(os.path.dirname(pick[0]))}/"
              f"{os.path.basename(pick[0])}")
        tea = Teacher(spec, JOINTS, torch.zeros(nj), pick[0], DEV)
        cmd4 = torch.tensor([[1.0, 0.0, 0.0, 0.30]]).repeat(B, 1)
        o_t = tea.build_obs(S, cmd4)
        assert o_t.shape == (B, 35), o_t.shape
        a_t = tea.act(S, cmd4)
        assert a_t.shape == (B, 6) and torch.isfinite(a_t).all(), a_t
        print(f"[teacher] obs={tuple(o_t.shape)} act 范围 [{float(a_t.min()):+.2f}, {float(a_t.max()):+.2f}]")
        # 竖着站 vs 侧翻，老师应该给出明显不同的动作
        S_up = (p, torch.eye(3).expand(B, 3, 3), S[2], S[3], S[4], S[5])
        R_side = quat_to_mat(torch.tensor([[0.7071, 0.0, 0.7071, 0.0]])).expand(B, 3, 3)
        S_side = (p, R_side, S[2], S[3], S[4], S[5])
        d = (tea.act(S_up, cmd4) - tea.act(S_side, cmd4)).abs().max()
        print(f"[teacher] 姿态变化引起的动作差 = {float(d):.3f}")
        assert float(d) > 1e-3, "老师对姿态不敏感，obs 拼接可能错了"

    print("\n[selftest] 全部通过")


if __name__ == "__main__":
    main()
