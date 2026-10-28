# SPDX-License-Identifier: MIT
"""检查点评测：在一组固定速度/角速度指令上各跑一次，报告归一化回报与跟踪误差。"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, default=None)
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--watch", action="store_true")
parser.add_argument("--every", type=int, default=1)
parser.add_argument("--steps", type=int, default=400, help="每条指令评测步数（50Hz -> 8 s）")
parser.add_argument("--height_cmd", type=float, default=None,
                    help="指令高度，默认 z_stand-0.04（不能叫 --height，会和 SimulationApp 参数冲突）")
parser.add_argument("--z_stand", type=float, default=None)
parser.add_argument("--term_height", type=float, default=None)
parser.add_argument("--settle_steps", type=int, default=120)
parser.add_argument("--friction", type=float, default=1.0)
parser.add_argument("--model_json", type=str, default=os.path.join(REPO, "data", "wheelbipe_v14_2_model.json"))
parser.add_argument("--no_gimbal_usd", type=int, default=0)
parser.add_argument("--no_kin", action="store_true")
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
from supermimic_wheelbipe.robot import WHEEL_RADIUS, build_spec, make_robot_cfg, usd_path  # noqa: E402


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = args.device
    usd = usd_path(no_gimbal=bool(args.no_gimbal_usd))
    sim_cfg = make_sim_cfg(dev, friction=args.friction)
    pairs = eval_pairs()
    N = len(pairs)
    sim, scene, robot = make_world(sim_cfg, N, make_robot_cfg(usd))
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
            print(f"[eval] FK 不可用（{e}）")
    z_stand, _ = resolve_stand_height(sim, scene, robot, spec, kin, default_q,
                                      settle_steps=args.settle_steps, override=args.z_stand)
    h = args.height_cmd if args.height_cmd is not None else default_height_cmd(z_stand)
    term_height = args.term_height if args.term_height is not None else default_term_height(z_stand)
    cmds = [[vx, 0.0, wz, h] for vx, wz in pairs]
    print(f"[eval] height={h:.3f} m (z_stand={z_stand:.3f})  term_height={term_height:.3f}")
    cmd = CommandLib.fixed(cmds, dev, frames=args.steps + 1)
    mdp = MDP(spec, cmd, kin=kin, default_q=default_q, cfg=RewardCfg(term_height=term_height))
    env = TrackingSim(sim, scene, robot, spec, cmd, mdp, eval_mode=True, z_stand=z_stand,
                      friction=args.friction, verbose=False)
    C = cmd.clip_len - 1  # 每条指令的可评步数
    print(f"[eval] {N} 条指令 x {int(C[0])} 步")

    @torch.no_grad()
    def evaluate(ck: dict) -> dict:
        a_ = ck["args"]
        K, Hh = a_["goal_frames"], a_["hist"]
        networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
        pi = Policy(mdp.obs_dim, spec.na, 4 * K, hist=Hh).to(dev)
        sd = ck["pi"]
        try:
            pi.load_state_dict(sd)
        except RuntimeError:
            # 允许维度不同的旧 checkpoint：只加载形状匹配的张量
            own = pi.state_dict()
            pi.load_state_dict({k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}, strict=False)
            print("[eval] 部分权重维度不匹配，已按形状过滤加载")
        pi.eval()
        ids = torch.arange(N, device=dev)
        env.done_eval[:] = False
        env.failed_eval[:] = False
        env.reset(ids, cmd.clip_start.clone())
        ret = torch.zeros(N, device=dev)
        e_lin = torch.zeros(N, device=dev)
        e_ang = torch.zeros(N, device=dev)
        e_h = torch.zeros(N, device=dev)
        cnt = torch.zeros(N, device=dev)
        hist = None
        while not bool(env.done_eval.all()):
            S = env.state()
            o = mdp.obs(S, env.a_prev)
            hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
            a = pi(hist, cmd.goal(env.f, K))
            alive = ~env.done_eval.clone()
            out = env.step(a)
            ret += torch.where(alive, out["r"], torch.zeros_like(out["r"]))
            t = out["terms"]
            e_lin += torch.where(alive, t["e_lin"].abs(), torch.zeros_like(e_lin))
            e_ang += torch.where(alive, t["e_ang"].abs(), torch.zeros_like(e_ang))
            e_h += torch.where(alive, t["e_height"].abs(), torch.zeros_like(e_h))
            cnt += alive.float()
        norm = ret / (C.float() * MAX_STEP_REWARD)
        cnt = cnt.clamp(min=1)
        res = {
            "norm_return": float(norm.mean()),
            "norm_return_framew": float(ret.sum() / (C.sum() * MAX_STEP_REWARD)),
            "n_failed": int(env.failed_eval.sum()),
            "failed": [cmd.names[i] for i in env.failed_eval.nonzero().flatten().tolist()],
            "mean_abs_e_lin": float((e_lin / cnt).mean()),
            "mean_abs_e_ang": float((e_ang / cnt).mean()),
            "mean_abs_e_h": float((e_h / cnt).mean()),
            "per_cmd": {cmd.names[i]: round(float(norm[i]), 3) for i in range(N)},
        }
        return res

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=dev)
        print(json.dumps({k: v for k, v in evaluate(ck).items() if k != "per_cmd"}, indent=1))
        return

    assert args.run_dir, "需要 --ckpt 或 --run_dir"
    out_csv = os.path.join(args.run_dir, "eval.csv")
    done = set()
    if os.path.exists(out_csv):
        done = {int(l.split(",")[0]) for l in open(out_csv).read().strip().splitlines()[1:] if l.strip()}
    else:
        with open(out_csv, "w") as f:
            f.write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,"
                    "mean_abs_e_lin,mean_abs_e_ang,mean_abs_e_h,failed\n")
    while True:
        ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))
        todo = [c for c in ckpts if int(os.path.basename(c)[5:11]) not in done]
        todo = [c for c in todo if (int(os.path.basename(c)[5:11]) // 1000) % args.every == 0]
        for c in todo:
            it = int(os.path.basename(c)[5:11])
            ck = torch.load(c, map_location=dev)
            t0 = time.time()
            res = evaluate(ck)
            line = (f"{it},{ck['wall_time'] / 3600:.4f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},"
                    f"{res['norm_return']:.4f},{res['norm_return_framew']:.4f},{res['n_failed']},"
                    f"{res['mean_abs_e_lin']:.4f},{res['mean_abs_e_ang']:.4f},{res['mean_abs_e_h']:.4f},"
                    f"{'|'.join(res['failed'])}\n")
            with open(out_csv, "a") as f:
                f.write(line)
            json.dump(res, open(c.replace(".pt", "_eval.json"), "w"), indent=1)
            print(f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm {res['norm_return']:.4f}, "
                  f"failed {res['n_failed']}/{N}, |e_lin| {res['mean_abs_e_lin']:.3f} "
                  f"|e_ang| {res['mean_abs_e_ang']:.3f} |e_h| {res['mean_abs_e_h']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            done.add(it)
        if not args.watch:
            break
        time.sleep(60)


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
