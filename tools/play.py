# SPDX-License-Identifier: MIT
"""回放 / 录视频：一条命令跑多个机器人（不同速度指令），拼成 2x2 画面存 mp4。"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
# 策略来源（二选一）
parser.add_argument("--ckpt", type=str, default=None, help="SuperMimic 训练出的 ckpt_*.pt")
parser.add_argument("--teacher", type=str, default=None, help="预训练 PPO：auto 或 run 目录")
# 播放内容
parser.add_argument("--mode", choices=["grid", "single"], default="grid")
parser.add_argument("--cmd", type=float, nargs=3, default=(1.0, 0.0, -1.0), metavar=("VX", "WZ", "H"),
                    help="single 模式的指令；H<=0 自动取 z_stand-0.04")
parser.add_argument("--commands", type=str, default="0.5,1.0,1.5,-0.5", help="grid 模式的 vx，逗号分隔，1~4 个")
parser.add_argument("--wzs", type=str, default=None, help="grid 模式的 wz，与 --commands 一一对应")
parser.add_argument("--height_cmd", type=float, default=None,
                    help="指令高度，默认 z_stand-0.04（不能叫 --height，会和 SimulationApp 冲突）")
parser.add_argument("--steps", type=int, default=300, help="每条指令录多少控制步（50Hz，300=6s）")
parser.add_argument("--stagger", type=int, default=0, help="每条 clip 开头额外空跑的步数（错开相位用）")
# 视频
parser.add_argument("--video", type=str, default="videos/play_grid.mp4",
                    help="输出 mp4；写 none 则只开窗口看")
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--resolution", type=int, nargs=2, default=(640, 360), metavar=("W", "H"))
parser.add_argument("--cam", choices=["follow", "static"], default="follow")
parser.add_argument("--cam_offset", type=float, nargs=3, default=(2.6, -2.6, 1.3), metavar=("DX", "DY", "DZ"))
parser.add_argument("--focal", type=float, default=20.0)
parser.add_argument("--stills", action="store_true", help="另外把每格最后一帧存成 png，方便贴汇报")
parser.add_argument("--label", type=int, default=1, help="1=在每格左上角烙上速度指令")
parser.add_argument("--title", type=str, default=None, help="整幅画面顶部的分段标题（中文 OK）")
parser.add_argument("--leg_lo", type=float, default=None, help="变腿模式：最低高度 (m)")
parser.add_argument("--leg_hi", type=float, default=None, help="变腿模式：最高高度 (m)，给了就按正弦周期变腿")
parser.add_argument("--leg_period", type=float, default=2.0, help="变腿周期 (s)")
parser.add_argument("--leg_phase", type=float, default=0.0, help="变腿相位偏移 (s)")
# 环境
parser.add_argument("--friction", type=float, default=1.0)
parser.add_argument("--z_stand", type=float, default=None)
parser.add_argument("--settle_steps", type=int, default=120)
parser.add_argument("--term_height", type=float, default=None)
parser.add_argument("--no_gimbal_usd", type=int, default=0)
parser.add_argument("--no_kin", action="store_true")
parser.add_argument("--model_json", type=str, default=os.path.join(REPO, "data", "wheelbipe_v14_2_model.json"))
# 策略超参（需与训练一致；ckpt 里的 args 优先）
parser.add_argument("--goal_frames", type=int, default=10)
parser.add_argument("--hist", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.ckpt is None and args.teacher is None:
    parser.error("需要 --ckpt 或 --teacher 之一")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import supermimic_wheelbipe.networks as networks  # noqa: E402
from supermimic_wheelbipe.command import CommandLib  # noqa: E402
from supermimic_wheelbipe.constants import default_height_cmd, default_term_height  # noqa: E402
from supermimic_wheelbipe.env import TrackingSim, make_sim_cfg, make_world, resolve_stand_height  # noqa: E402
from supermimic_wheelbipe.kinematics import Kinematics  # noqa: E402
from supermimic_wheelbipe.mdp import MAX_STEP_REWARD, MDP, RewardCfg  # noqa: E402
from supermimic_wheelbipe.model import ensure_model, wheel_bodies  # noqa: E402
from supermimic_wheelbipe.networks import Policy  # noqa: E402
from supermimic_wheelbipe.robot import CONTROL_DT, build_spec, make_robot_cfg, usd_path  # noqa: E402
from supermimic_wheelbipe.teacher import Teacher, pick_best_teacher  # noqa: E402

CAM_PRIM = "/World/rec_cam"
LIVE = args.video is None or str(args.video).lower() in ("none", "off", "-")


# 抓帧
class Recorder:
    """用 Replicator 的 render product + rgb annotator 抓帧（不依赖 viewport）。"""

    def __init__(self, resolution, focal, sim=None):
        from pxr import Gf, UsdGeom

        import omni.replicator.core as rep
        import omni.usd

        self.rep = rep
        self.sim = sim
        self.Gf = Gf
        stage = omni.usd.get_context().get_stage()
        prim = stage.DefinePrim(CAM_PRIM, "Camera")
        cam = UsdGeom.Camera(prim)
        cam.GetFocalLengthAttr().Set(float(focal))
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 200.0))
        self.xform = UsdGeom.Xformable(prim)
        self.xform.ClearXformOpOrder()
        self.op = self.xform.AddTransformOp()
        self.set_pose((3.0, -3.0, 1.5), (0.0, 0.0, 0.30))
        self.rp = rep.create.render_product(CAM_PRIM, (int(resolution[0]), int(resolution[1])))
        # 照抄 Isaac Lab ManagerBasedEnv.render()：device="cpu" + attach([rp])，
        # 取帧前调 sim.render()，headless 下用 rep.orchestrator.step() 会卡死
        self.ann = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.ann.attach([self.rp])
        print(f"[rec] 抓帧后端 replicator(cpu): {CAM_PRIM} @ {tuple(resolution)}", flush=True)

    def set_pose(self, eye, lookat, up=(0.0, 0.0, 1.0)):
        Gf = self.Gf
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*lookat), Gf.Vec3d(*up))
        self.op.Set(view.GetInverse())

    def frame(self):
        if self.sim is not None:
            self.sim.render()  # 渲染一帧（不推进物理）
        a = np.asarray(self.ann.get_data())
        if a.size == 0:
            return None
        if a.dtype != np.uint8:
            a = np.clip(a * (255.0 if float(a.max()) <= 1.5 else 1.0), 0, 255).astype(np.uint8)
        return a[..., :3]

    def close(self):
        try:
            self.ann.detach()
            self.rp.destroy()
        except Exception:  # noqa: BLE001
            pass


# 策略
class PolicyRunner:
    """统一封装：SuperMimic 策略 / 预训练老师，对外都是 (S, frame, hist) -> (action, hist)。"""

    def __init__(self, env, mdp, spec, joint_names, default_q, dev):
        self.env, self.mdp, self.spec, self.dev = env, mdp, spec, dev
        self.kind = "teacher" if args.teacher else "ckpt"
        self.hist = None
        if self.kind == "teacher":
            self.tea = Teacher(spec, list(joint_names), default_q, pick_best_teacher(args.teacher), dev)
        else:
            self.load_ckpt(args.ckpt, dev)

    def load_ckpt(self, path, dev):
        ck = torch.load(path, map_location=dev)
        meta = ck.get("args", {})
        self.K = int(meta.get("goal_frames", args.goal_frames))
        self.Hh = int(meta.get("hist", args.hist))
        networks.AMP_DTYPE = torch.bfloat16 if meta.get("amp", "none") == "bf16" else None
        self.pi = Policy(self.mdp.obs_dim, self.spec.na, 4 * self.K, hist=self.Hh).to(dev)
        try:
            self.pi.load_state_dict(ck["pi"])
        except RuntimeError:
            own = self.pi.state_dict()
            self.pi.load_state_dict({k: v for k, v in ck["pi"].items() if k in own and own[k].shape == v.shape},
                                    strict=False)  # fmt: skip
            print("[play] 部分权重维度不匹配，已按形状过滤加载")
        self.pi.eval()
        print(f"[play] ckpt iteration {ck.get('iteration')}  goal_frames={self.K} hist={self.Hh}")

    def reset(self, env_ids=None):
        self.hist = None
        if self.kind == "teacher":
            self.tea.reset(env_ids)

    @torch.no_grad()
    def act(self, S, f):
        if self.kind == "teacher":
            return self.tea.act(S, self.env.cmd.cmd_at(f))
        o = self.mdp.obs(S, self.env.a_prev)
        self.hist = o.unsqueeze(1).repeat(1, self.Hh, 1) if self.hist is None else torch.cat(
            (self.hist[:, 1:], o.unsqueeze(1)), 1
        )
        return self.pi(self.hist, self.env.cmd.goal(f, self.K))

    def action_semantics(self) -> str:
        return "teacher" if self.kind == "teacher" else "ckpt"


# 主流程
def build(dev):
    usd = usd_path(no_gimbal=bool(args.no_gimbal_usd))
    sim, scene, robot = make_world(make_sim_cfg(dev, friction=args.friction), 1, make_robot_cfg(usd))
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
            print(f"[play] FK 不可用（{e}）")
    z_stand, _ = resolve_stand_height(sim, scene, robot, spec, kin, default_q,
                                      settle_steps=args.settle_steps, override=args.z_stand)
    return sim, scene, robot, spec, kin, z_stand, default_q


def parse_cmds():
    if args.mode == "single":
        return [(float(args.cmd[0]), float(args.cmd[1]), float(args.cmd[2]))]
    vx = [float(v) for v in args.commands.split(",") if v.strip()]
    wz = [0.0] * len(vx) if args.wzs is None else [float(v) for v in args.wzs.split(",") if v.strip()]
    assert len(wz) == len(vx), f"--wzs({len(wz)}) 与 --commands({len(vx)}) 数量不一致"
    assert 1 <= len(vx) <= 4, "grid 模式支持 1~4 条指令"
    return [(v, w, float(args.cmd[2])) for v, w in zip(vx, wz)]


def draw_label(frame, vx, wz, h, extra=""):
    """在帧左上角写上这条片段的速度指令（用于对比不同速度的视频）。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:  # noqa: BLE001
        return frame
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    hh, ww = frame.shape[:2]
    size = max(11, hh // 13)
    font = None
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:  # noqa: BLE001
                pass
    if font is None:
        try:
            font = ImageFont.load_default(size=size)
        except Exception:  # noqa: BLE001
            font = ImageFont.load_default()
    lines = [f"vx = {vx:+.1f} m/s"]
    if abs(wz) > 1e-6:
        lines.append(f"wz = {wz:+.1f} rad/s")
    lines.append(f"h = {h:.3f} m" + ("  " + extra if extra else ""))
    pad = max(3, size // 4)
    y = pad
    for txt in lines:
        try:
            box = d.textbbox((0, 0), txt, font=font)
            tw, th = box[2] - box[0], box[3] - box[1]
        except Exception:  # noqa: BLE001
            tw, th = len(txt) * size // 2, size
        d.rectangle([pad - 1, y - 1, pad + tw + 4, y + th + 4], fill=(0, 0, 0, 150))
        d.text((pad + 2, y), txt, font=font, fill=(255, 255, 60, 255))
        y += th + pad + 3
    return np.asarray(img)[..., :3]


def draw_title(frame, text):
    """在整幅画面顶部画一条中文标题带（用于 B 站展示片分段）。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:  # noqa: BLE001
        return frame
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    hh, ww = frame.shape[:2]
    size = max(20, int(hh / 15))
    font = None
    for path, idx in (
        ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc", 0),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
    ):
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size, index=idx)
                break
            except Exception:  # noqa: BLE001
                pass
    if font is None:
        try:
            font = ImageFont.load_default(size=size)
        except Exception:  # noqa: BLE001
            font = ImageFont.load_default()
    band = size + 12
    d.rectangle([0, 0, ww, band], fill=(0, 0, 0, 185))
    try:
        box = d.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        ox, oy = box[0], box[1]
    except Exception:  # noqa: BLE001
        tw, th, ox, oy = len(text) * size // 2, size, 0, 0
    d.text(((ww - tw) // 2 - ox, (band - th) // 2 - oy), text, font=font, fill=(255, 214, 60, 255))
    return np.asarray(img)[..., :3]


def tile_to_grid(clips, res):
    """把 1~4 段等长片段拼成 1x1 / 1x2 / 2x2 画布。"""
    n = min(len(c) for c in clips)
    h, w = res[1], res[0]
    if len(clips) == 1:
        return [clips[0][i] for i in range(n)]
    cols = 2
    rows = 1 if len(clips) <= 2 else 2
    out = []
    for i in range(n):
        canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for k, frames in enumerate(clips[:4]):
            r, c = divmod(k, cols)
            canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = frames[i]
        out.append(canvas)
    return out


def save_video(path, frames, fps):
    """优先 imageio 存 mp4；失败退化成 png 帧序列（返回实际产物路径）。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import imageio

        try:
            imageio.mimsave(path, frames, fps=fps, quality=8)
        except TypeError:  # 某些版本不接受 quality
            imageio.mimsave(path, frames, fps=fps)
        return path
    except Exception as e:  # noqa: BLE001
        fdir = path.replace(".mp4", "_frames")
        os.makedirs(fdir, exist_ok=True)
        try:
            from PIL import Image

            for k, f in enumerate(frames):
                Image.fromarray(f).save(os.path.join(fdir, f"frame_{k:05d}.png"))
        except Exception as e2:  # noqa: BLE001
            raise RuntimeError(f"imageio({e}) 与 PIL({e2}) 都存不了，请 pip install imageio imageio-ffmpeg") from e2
        print(f"[rec] 存 mp4 失败（{e}），已写成 png 帧序列")
        print(f"[rec] 合成: ffmpeg -framerate {fps} -i {fdir}/frame_%05d.png -pix_fmt yuv420p out.mp4")
        return fdir


def main():
    dev = args.device
    cmds = parse_cmds()
    sim, scene, robot, spec, kin, z_stand, default_q = build(dev)
    if args.height_cmd is not None:
        h_cmd = args.height_cmd
    elif args.mode == "single" and float(args.cmd[2]) > 0:
        h_cmd = float(args.cmd[2])
    else:
        h_cmd = default_height_cmd(z_stand)
    term_height = args.term_height if args.term_height is not None else default_term_height(z_stand)
    T = args.steps + 64
    h_frames = None
    if args.leg_hi is not None:  # 变腿模式：高度按正弦周期变化，4 格错开相位
        import math

        lo = float(args.leg_lo if args.leg_lo is not None else max(z_stand - 0.03, 0.14))
        hi = float(args.leg_hi)
        per = max(0.4, float(args.leg_period)) / CONTROL_DT
        mid, amp = 0.5 * (lo + hi), 0.5 * (hi - lo)
        tt = torch.arange(T, device=dev, dtype=torch.float32)
        rows, h_frames = [], []
        for i, (v, w, _) in enumerate(cmds):
            ph = float(args.leg_phase) / CONTROL_DT + i * per / max(len(cmds), 1)
            hh = mid + amp * torch.sin(2.0 * math.pi * (tt + ph) / per)
            rows.append(torch.stack([torch.full((T,), v, device=dev), torch.zeros(T, device=dev),
                                     torch.full((T,), w, device=dev), hh], -1))  # fmt: skip
            h_frames.append(hh)
        cmd = CommandLib(torch.cat(rows, 0), [T] * len(cmds), dev,
                         names=[f"vx{v:+.1f}_leg{lo:.2f}-{hi:.2f}" for v, _w, _ in cmds])  # fmt: skip
        print(f"[play] 变腿模式: h {lo:.2f}↔{hi:.2f} m，周期 {float(args.leg_period):.1f}s，"
              f"4 格相位错开 {float(args.leg_period) / max(len(cmds), 1):.1f}s")
    else:
        cmd = CommandLib.fixed([[v, 0.0, w, h_cmd] for v, w, _ in cmds], dev, frames=T)
    mdp = MDP(spec, cmd, kin=kin, default_q=default_q, cfg=RewardCfg(term_height=term_height))
    env = TrackingSim(sim, scene, robot, spec, cmd, mdp, eval_mode=True, reset_noise=False,
                      z_stand=z_stand, friction=args.friction, verbose=False)  # fmt: skip
    env.render_physics = False  # 物理只由 step() 推进；抓帧用 Recorder.frame() 里的 sim.render()

    run = PolicyRunner(env, mdp, spec, robot.joint_names, default_q, dev)
    print(f"[play] 来源={run.action_semantics()}  z_stand={z_stand:.3f}  height_cmd={h_cmd:.3f}  "
          f"term_height={term_height:.3f}")
    print(f"[play] 指令: {[f'vx={v:+.2f} wz={w:+.2f}' for v, w, _ in cmds]}  x {args.steps} 步 "
          f"({args.steps * CONTROL_DT:.1f}s)")

    rec = None if LIVE else Recorder(args.resolution, args.focal, sim)
    if rec is not None:  # 预热两帧，避免第一帧是黑的
        for _ in range(2):
            rec.frame()
    ids = torch.tensor([0], device=dev)
    clips, stats = [], []
    for i, (vx, wz, _) in enumerate(cmds):
        env.reset(ids, cmd.clip_start[i:i + 1])
        run.reset(ids)
        clip, ret, n, died = [], 0.0, 0, None
        for _ in range(max(args.stagger, 0)):
            env.step(torch.zeros(1, spec.na, device=dev))
        for t in range(args.steps):
            S = env.state()
            p = S[0][0].tolist()
            if rec is not None and args.cam == "follow":
                rec.set_pose([p[0] + args.cam_offset[0], p[1] + args.cam_offset[1], args.cam_offset[2]],
                             [p[0], p[1], 0.25])  # fmt: skip
            if LIVE:
                sim.set_camera_view(eye=[p[0] + args.cam_offset[0], p[1] + args.cam_offset[1], args.cam_offset[2]],
                                    target=[p[0], p[1], 0.25])  # fmt: skip
            a = run.act(S, env.f)
            out = env.step(a)
            ret += float(out["r"].mean())
            n += 1
            if rec is not None:
                f = rec.frame()
                if f is not None:
                    clip.append(f)
            if died is None and bool(env.done_eval.any()):
                died = t
        if rec is not None and clip and len(clip) < args.steps:  # 摔了就冻结最后一帧，保证四格对齐
            clip += [clip[-1]] * (args.steps - len(clip))
        clips.append(clip)
        stats.append((vx, wz, died, ret / max(n, 1) / MAX_STEP_REWARD))
        print(f"[play] 片段 {i + 1}/{len(cmds)}: vx={vx:+.2f} wz={wz:+.2f} "
              f"{'存活' if died is None else f'第 {died} 步终止'}  "
              f"平均步回报 {ret / max(n, 1) / MAX_STEP_REWARD:.3f}", flush=True)

    if rec is not None:
        rec.close()
    print("[play] 汇总: " + " | ".join(
        f"vx{vx:+.2f}{'' if d is None else f'(摔在{d})'} 回报{r:.2f}" for vx, _w, d, r in stats))
    if rec is None:
        print("[play] 只看了窗口，没有录视频（--video none）")
        return
    if not any(clips):
        print("[play] 一帧都没抓到：确认启动参数里带了 --enable_cameras；"
              "若仍为空，看 stderr 里 Replicator 的报错（可能是渲染器没起来）")
        return

    if args.label:  # 在每格左上角烙上速度/高度指令
        for i, clip in enumerate(clips):
            if not clip:
                continue
            hv = h_frames[i] if h_frames is not None else None
            out = []
            for k, f in enumerate(clip):
                if hv is not None:
                    h_now = float(hv[min(k, len(hv) - 1)])
                else:
                    # cmds[i][2] 是命令行里显式给的高度；<=0 表示"用默认值"
                    h_now = float(cmds[i][2]) if float(cmds[i][2]) > 0 else h_cmd
                out.append(draw_label(f, cmds[i][0], cmds[i][1], h_now,
                                      extra="(leg cycle)" if hv is not None else ""))  # fmt: skip
            clips[i] = out
    grid = tile_to_grid(clips, args.resolution)
    if args.title:  # 分段标题（B 站展示片用）
        grid = [draw_title(f, args.title) for f in grid]
    out = os.path.abspath(os.path.expanduser(args.video))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    saved = save_video(out, grid, args.fps)
    if args.stills:
        still = os.path.join(os.path.dirname(saved), "stills")
        os.makedirs(still, exist_ok=True)
        for i, clip in enumerate(clips):
            if clip:
                try:
                    import imageio

                    imageio.imwrite(os.path.join(still, f"cmd{i}_vx{cmds[i][0]:+.2f}.png"), clip[-1])
                except Exception:  # noqa: BLE001
                    pass
    size = os.path.getsize(saved) / 1e6 if os.path.isfile(saved) else 0.0
    print(f"\n[rec] 写出 {saved}  ({len(grid)} 帧 @ {args.fps} fps, "
          f"{grid[0].shape[1]}x{grid[0].shape[0]}, {size:.1f} MB)")
    print(f"[rec] 拷回本地电脑（把 <用户>@<服务器> 换成你的）:\n"
          f"      scp <用户>@<服务器>:{saved}  .")


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
