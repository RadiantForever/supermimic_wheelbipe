# SPDX-License-Identifier: MIT
"""标定 / 自检：关节清单、FK 校验、站立高度、弹簧力、落地过程。"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model_json", type=str, default=os.path.join(REPO, "data", "wheelbipe_v14_2_model.json"))
parser.add_argument("--stretch", type=float, default=None, help="覆盖 z_stand")
parser.add_argument("--settle_steps", type=int, default=300)
parser.add_argument("--no_gimbal_usd", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

from supermimic_wheelbipe.command import CommandLib  # noqa: E402
from supermimic_wheelbipe.env import TrackingSim, make_sim_cfg, make_world  # noqa: E402
from supermimic_wheelbipe.kinematics import Kinematics  # noqa: E402
from supermimic_wheelbipe.mdp import MDP, normalize_xi  # noqa: E402
from supermimic_wheelbipe.model import ensure_model, wheel_bodies  # noqa: E402
from supermimic_wheelbipe.robot import (  # noqa: E402
    SPRING_FORCE_DOWN,
    SPRING_FORCE_UP,
    SPRING_LENGTH,
    SPRING_OFFSET,
    WHEEL_RADIUS,
    build_spec,
    make_robot_cfg,
    spring_force,
    usd_path,
)


def _t(v):
    return v.torch if hasattr(v, "torch") else v


def main():
    dev = args.device
    usd = usd_path(no_gimbal=bool(args.no_gimbal_usd))
    print(f"[calib] USD = {usd}")
    sim, scene, robot = make_world(make_sim_cfg(dev), 1, make_robot_cfg(usd))

    print("\n=== 1. articulation 清单 ===")
    print(f"body_names ({len(robot.body_names)}): {list(robot.body_names)}")
    print(f"joint_names ({len(robot.joint_names)}):")
    for i, n in enumerate(robot.joint_names):
        print(f"  [{i:2d}] {n}")

    lim = _t(robot.data.joint_pos_limits)[0]
    spec = build_spec(robot.joint_names, q_lo=lim[:, 0].clone(), q_hi=lim[:, 1].clone())
    print("\n=== 2. 分组 ===")
    print(spec.describe())
    print(f"动作维 na={spec.na}  状态维 nq={spec.nq}+nd={spec.nd}")

    print("\n=== 3. 弹簧模型 ===")
    print(f"spring2 = {[spec.joint_names[i] for i in spec.spring_idx]}")
    q0 = torch.zeros(len(spec.spring_idx), device=dev)
    print(f"q_spring=0 -> 每根弹簧 {float(spring_force(q0[:1], torch.zeros(1, device=dev))[0]):.1f} N "
          f"(offset={SPRING_OFFSET} m, {SPRING_FORCE_DOWN}->{SPRING_FORCE_UP} N / {SPRING_LENGTH} m)")
    if len(spec.spring_idx) == 0:
        print("!! 没有找到 spring2 关节：腿会塌，检查关节命名")

    # FK 校验
    print("\n=== 4. FK 校验（USD 树 vs 仿真）===")
    kin = None
    try:
        model = ensure_model(usd, path=args.model_json)
        wb = wheel_bodies(model, spec)
        kin = Kinematics(model, list(robot.joint_names), dev, tracked=list(robot.body_names), wheel_bodies=wb)
        print(f"运动学树: {kin.num_bodies} 刚体 / {kin.num_movable} 可动; 轮子刚体={wb}")
        nj = len(robot.joint_names)
        q = (torch.rand(1, nj, device=dev) - 0.5) * 0.6
        qd = torch.zeros_like(q)
        org = scene.env_origins[0:1]
        root_p = org.clone()
        root_p[:, 2] = 0.6
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev)
        robot.write_root_pose_to_sim(torch.cat((root_p, quat), -1), env_ids=torch.tensor([0], device=dev))
        robot.write_joint_state_to_sim(q, qd, None, torch.tensor([0], device=dev))
        scene.update(0.0)
        sp = _t(robot.data.body_link_pose_w)[0, :, :3] - org
        R = torch.eye(3, device=dev).unsqueeze(0)
        bp = kin.forward(root_p - org, R, torch.zeros(1, 3, device=dev), torch.zeros(1, 3, device=dev), q, qd)[0][0]
        err = (bp - sp).norm(dim=-1)
        print(f"逐刚体位置误差: max {float(err.max()):.2e} m, mean {float(err.mean()):.2e} m")
        worst = int(err.argmax())
        print(f"最差刚体: {kin.tracked_names[worst]}  误差 {float(err[worst]):.2e} m")
        if float(err.max()) > 2e-2:
            print("!! FK 与仿真不一致（>2cm）：no_fork 奖励会算错。"
                  "先跑 tools/extract_model.py 看树结构，必要时 --no_kin 关掉该项。")
    except Exception as e:  # noqa: BLE001
        print(f"FK 不可用: {e}")

    # 站立高度
    print("\n=== 5. 站立高度 ===")
    default_q = _t(robot.data.default_joint_pos)[0].clone()
    z_fk = None
    if kin is not None:
        z_fk = kin.rest_height(default_q, WHEEL_RADIUS, kin.wheel_names)
        print(f"FK 估计（全零关节、轮子触地）= {z_fk:.4f} m")
    z_stand = args.stretch if args.stretch is not None else (z_fk if z_fk is not None else 0.38)
    print(f"使用 z_stand = {z_stand:.4f} m")

    # 静置过程
    print("\n=== 6. 零动作静置（加速度/弹簧是否撑得住）===")
    cmd = CommandLib.fixed([[0.0, 0.0, 0.0, 0.26]], dev, frames=args.settle_steps + 10)
    mdp = MDP(spec, cmd, kin=kin, default_q=default_q)
    env = TrackingSim(sim, scene, robot, spec, cmd, mdp, eval_mode=True, z_stand=z_stand,
                      friction=1.0, verbose=False)
    env.reset(torch.tensor([0], device=dev), cmd.clip_start.clone())
    na = spec.na
    hdr = "  step        z      wheel_z   spring2(2)          leg_act(4)"
    print(hdr)
    for i in range(args.settle_steps):
        env.apply_action(torch.zeros(1, na, device=dev))
        env.physics()
        if i % 25 == 0 or i == args.settle_steps - 1:
            S = env.state()
            q = S[4]
            sp = q[:, [mdp.q_pos[j] for j in spec.spring_idx]].reshape(-1)
            la = q[:, mdp.leg_act_in_q].reshape(-1)
            wl = [i for i, n in enumerate(kin.tracked_names) if n in kin.wheel_names] if kin else []
            wz = 0.0
            if kin is not None and wl:
                P = kin.forward(S[0], S[1], S[2], S[3], *mdp.full_joints(S[4], S[5]))[0][0]
                wz = float(P[wl, 2].min())
            print(f"  {i:4d}  {float(S[0][0,2]):7.4f}  {wz:8.4f}   "
                  f"{' '.join(f'{float(v):+.4f}' for v in sp):<20} {' '.join(f'{float(v):+.3f}' for v in la)}")

    print("\n结论建议：")
    print("  * 若静置后 z 基本不变且 wheel_z≈轮半径 -> 弹簧/高度参数可用；")
    print("  * 若 z 持续下降到 <0.2 或轮子不动而机身下沉 -> 弹簧力方向/大小需要调；")
    print(f"  * 把 RewardCfg.term_height 设为略低于稳态高度（例如 {max(z_stand - 0.12, 0.10):.2f}）；")
    print("  * 指令 height 范围建议取 [稳态高度-0.10, 稳态高度] 并留 0.02 余量。")


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
