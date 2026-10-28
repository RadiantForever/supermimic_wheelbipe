# SPDX-License-Identifier: MIT
"""在 MuJoCo（纯 CPU）里跑 SuperMimic 训练出来的策略，键盘遥控轮腿机器人。"""

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
from supermimic_wheelbipe.gamepad import BUTTONS as PAD_BUTTONS  # noqa: E402
from supermimic_wheelbipe.gamepad import GamepadCmd, XInputPad, describe as pad_desc  # noqa: E402
from supermimic_wheelbipe.gamepad import selftest as pad_selftest  # noqa: E402
from supermimic_wheelbipe.mdp import MDP  # noqa: E402
from supermimic_wheelbipe.mujoco_env import DEFAULT_XML, MujocoWheelbipe, load_policy_ckpt  # noqa: E402

VX_LIM = (-1.5, 1.5)
WZ_LIM = (-10.21, 10.21)
H_LIM = (0.16, 0.28)                 # 0.16~0.20 只给训练过这个区间的权重（如蒸馏版 wb_c3）
VX_RAMP, WZ_RAMP, H_RAMP = 0.02, 0.06, 0.002      # 每个控制步（50 Hz）→ 1 m/s²、3 rad/s²、0.1 m/s
VX_SLEW, WZ_SLEW = 0.04, 0.2                      # 下发指令的变化率上限（2 m/s²、10 rad/s²）
TERM_COS = float(np.cos(np.deg2rad(40.0)))

HELP = """\
  W/S 或 ↑/↓  前进/后退       A/D 或 ←/→  左转/右转
  Q/E        变高/变低        空格        急停
  1/2/3      0.3/0.8/1.4 m/s   R          重置
  Esc        退出

  手柄（插上即自动接管 vx / wz，键盘仍可改高度）：
  左杆上下  速度（推杆幅度线性）   右杆左右  小陀螺（幅度线性）
  X         抬腿：高度预设循环 0.18→0.20→0.22→0.24
  十字键↑↓  高度 ±0.02            RT / LT   高度 连续 + / -
  右杆上下  高度 连续 ±0.12 m/s    RB / LB   高度 ±0.02
  A 急停（杆回中后再推解锁）  B 重置  START 退出
"""


class KeyboardCmd:
    """pynput 全局键盘 → (vx, wz, h) 目标。没有键盘依赖时退化成"恒定指令"。"""

    def __init__(self, vx: float, wz: float, h: float, enabled: bool = True):
        self.vx, self.wz, self.h = float(vx), float(wz), float(h)
        self.quit = False
        self.reset = False
        self._pressed: set[str] = set()
        self._listener = None
        self.enabled = enabled
        if enabled:
            self._start()

    def _start(self):
        try:
            from pynput import keyboard
        except Exception as e:  # noqa: BLE001
            print(f"[keys] pynput 不可用（{e}），只能跑恒定指令")
            self.enabled = False
            return

        def norm(k):
            try:
                if key_code := getattr(k, "char", None):
                    return key_code.lower()
            except Exception:  # noqa: BLE001
                pass
            name = getattr(k, "name", None)
            return {"up": "w", "down": "s", "left": "a", "right": "d"}.get(name, name)

        def on_press(k):
            n = norm(k)
            if n:
                self._pressed.add(n)

        def on_release(k):
            n = norm(k)
            self._pressed.discard(n)

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()

    def update(self):
        """每个控制步调用：按住即按斜坡改目标。"""
        if not self.enabled:
            return
        p = self._pressed
        if "w" in p:
            self.vx += VX_RAMP
        if "s" in p:
            self.vx -= VX_RAMP
        if "a" in p:
            self.wz += WZ_RAMP
        if "d" in p:
            self.wz -= WZ_RAMP
        if "q" in p:
            self.h += H_RAMP
        if "e" in p:
            self.h -= H_RAMP
        if " " in p or "space" in p:
            self.vx = self.wz = 0.0
        if "1" in p:
            self.vx = 0.3
        if "2" in p:
            self.vx = 0.8
        if "3" in p:
            self.vx = 1.4
        if "r" in p:
            self.reset = True
            p.discard("r")
        if "esc" in p:
            self.quit = True
        self.vx = float(np.clip(self.vx, *VX_LIM))
        self.wz = float(np.clip(self.wz, *WZ_LIM))
        self.h = float(np.clip(self.h, *H_LIM))

    def stop(self):
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:  # noqa: BLE001
                pass


def fmt_cmd(c):
    return f"vx={c.vx:+.2f} wz={c.wz:+.2f} h={c.h:.3f}"


def main():
    ap = argparse.ArgumentParser(description="MuJoCo 里用键盘遥控 SuperMimic 轮腿策略")
    ap.add_argument("--ckpt", type=str,
                    default=os.path.join("..", "export", "wb_export", "policies", "wb_final__last.pt"),
                    help="SuperMimic 检查点；--teacher 时只用来读 spec")
    ap.add_argument("--xml", type=str, default=DEFAULT_XML)
    ap.add_argument("--cmd", type=float, nargs=3, default=(0.0, 0.0, 0.20), metavar=("VX", "WZ", "H"),
                    help="初始指令；H 是绝对根高，wb_final 实测 0.20 最准（训练范围 0.20~0.28）")
    ap.add_argument("--steps", type=int, default=0, help="headless 模式跑多少控制步（0=一直跑）")
    ap.add_argument("--headless", action="store_true", help="不开窗口（无键盘，按 --cmd 恒定指令）")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--camera", type=str, default="track",
                    help="track（跟随）/ fixed / free")
    ap.add_argument("--trim", type=float, default=0.6,
                    help="外环速度修正增益（1/s，0=关闭）。策略看不到机身速度，换引擎后稳态会偏；"
                         "这个慢积分把实测速度补回指令上（限幅 ±0.4 m/s）")
    ap.add_argument("--gamepad", type=str, default="auto", choices=["auto", "off"],
                    help="auto=插上 XInput 手柄就接管 vx/wz；off=只认键盘")
    ap.add_argument("--pad_deadzone", type=float, default=0.08, help="摇杆死区（0~1）")
    ap.add_argument("--pad_expo", type=float, default=0.0,
                    help="摇杆曲线指数偏置：0=纯线性（默认），0.5=中间更细腻")
    ap.add_argument("--wz_ramp", type=float, default=0.0,
                    help="右杆左右 -> 偏航**爬升率**（rad/s²）。0=直接线性给转速；"
                         "设成 12 就是'推住杆、小陀螺线性起来'")
    ap.add_argument("--pad_probe", type=float, default=0.0,
                    help="只测手柄：打印 N 秒实时杆值（没插手柄时跑合成自检）后退出")
    ap.add_argument("--teacher", type=str, default=None,
                    help="用预训练 PPO 代替 SuperMimic 策略（auto 或 run 目录）。"
                         "它在 MuJoCo 里跟踪误差只有 0.03~0.04 m/s，比 wb_final 准得多")
    a = ap.parse_args()

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    networks.AMP_DTYPE = None

    # 手柄自检（不加载模型，插不上也能看响应曲线）
    if a.pad_probe > 0:
        pad = XInputPad()
        print(pad_desc(pad))
        if not pad.ok:
            return 1
        if pad.poll() is None:
            print("没检测到手柄 —— 下面用合成杆值验证映射曲线：")
            pad_selftest()
            print("\n接上手柄后（XInput 模式：Xbox 系 / 8BitDo 的 X 档 / PS 手柄经 DS4Windows）"
                  "再跑一次 --pad_probe 5 就能看到实时杆值。")
            return 0
        print(f"读手柄 {a.pad_probe:.0f}s（动一动杆和按键）：")
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < a.pad_probe:
            st = pad.poll()
            if st is None:
                print("  手柄断开")
                break
            held = ",".join(k for k, v in PAD_BUTTONS.items() if st["buttons"] & v)
            print(f"  lx={st['lx']:+.2f} ly={st['ly']:+.2f} rx={st['rx']:+.2f} "
                  f"ry={st['ry']:+.2f} lt={st['lt']:.2f} rt={st['rt']:.2f}  {held}")
            time.sleep(0.1)
        return 0

    ck = a.ckpt if os.path.isabs(a.ckpt) else os.path.normpath(os.path.join(REPO, a.ckpt))
    pi, spec, args = load_policy_ckpt(ck, a.device)   # 即使 --teacher 也要它来拿 spec
    K, Hh = int(args["goal_frames"]), int(args["hist"])
    env = MujocoWheelbipe(spec, xml=a.xml)

    teacher = None
    if a.teacher:
        from supermimic_wheelbipe.teacher import Teacher, pick_best_teacher

        tck = pick_best_teacher(a.teacher)
        teacher = Teacher(spec, spec.joint_names, torch.zeros(spec.nj), tck, device=a.device)
        print(f"[teacher] {tck}")
    else:
        print(f"[ckpt] {os.path.basename(ck)}  it={args['iteration']} "
              f"obs={spec.nq + spec.nd + 6 + spec.na} goal={4 * K} hist={Hh}")
    print(env.describe())

    # 站立高度：Isaac 的做法是"零动作静置后"的高度（resolve_stand_height），不是 FK 触地高度。
    # 直接用 FK 触地高度会多出 ~9 mm 的下落冲击，复位后头 0.3 s 的俯仰瞬态会直接把机器人顶翻。
    env.reset()
    z_contact = float(env.rest_height())
    env.reset(z=z_contact)
    for _ in range(150):
        env.set_zero_action()
        env.step()
    z_stand = float(env.base_z)
    term_h = z_stand - 0.06
    print(f"[stand] FK 触地 {z_contact:.4f} m / 静置 {z_stand:.4f} m，终止高度 {term_h:.4f} m")

    # goal 每帧 4 维，顺序必须是 (vx, vy, wz, height)
    cmd = CommandLib.fixed([[a.cmd[0], 0.0, a.cmd[1], a.cmd[2]]], device=a.device, frames=200000)
    mdp = MDP(spec, cmd, kin=None, default_q=torch.zeros(spec.nj), qd_limit=None)
    kb = KeyboardCmd(a.cmd[0], a.cmd[1], a.cmd[2], enabled=not a.headless)
    pad = None
    if a.gamepad != "off":
        pad = XInputPad()
        print(pad_desc(pad))
    keys = GamepadCmd(kb, pad, enabled=(not a.headless and pad is not None and pad.ok),
                      vx_max=VX_LIM[1], wz_max=WZ_LIM[1], h_rate=H_RAMP * 50 * 1.2,
                      h_lo=H_LIM[0], h_hi=H_LIM[1], deadzone=a.pad_deadzone,
                      expo=a.pad_expo, wz_ramp=a.wz_ramp)
    if not a.headless:
        print(HELP)
    print(f"[start] {fmt_cmd(keys)}" + ("   [手柄已接管]" if keys.connected else ""))

    viewer = None
    if not a.headless:
        try:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(env.m, env.d)
            if a.camera == "track":
                viewer.cam.type = 1  # mjCAMERA_TRACKING
                viewer.cam.trackbodyid = env.base_id
                viewer.cam.distance = 1.6
                viewer.cam.azimuth = 120.0
                viewer.cam.elevation = -18.0
        except Exception as e:  # noqa: BLE001
            print(f"[viewer] 打开窗口失败（{e}），退回 headless")
            viewer = None

    # 状态
    env.reset()
    hist = None
    a_prev = torch.zeros(1, spec.na, device=a.device)
    f = 0
    t_alive = 0.0
    n_reset = 0
    t0 = time.perf_counter()
    n = 0
    T = env.control_dt
    next_t = t0
    trim = 0.0          # 外环速度修正量（见 --trim）
    vx_cmd = 0.0        # 经过变化率限幅后真正下发的指令
    wz_cmd = 0.0

    def do_reset(msg=None):
        nonlocal hist, a_prev, f, t_alive, n_reset, trim, vx_cmd, wz_cmd
        env.reset(q_full=None, z=z_stand)
        hist, a_prev, f, t_alive, trim = None, torch.zeros(1, spec.na, device=a.device), 0, 0.0, 0.0
        vx_cmd = wz_cmd = 0.0        # 复位后从静止平滑起步，别给策略一个阶跃
        n_reset += 1
        if teacher is not None:
            teacher.reset()
        if msg:
            print(msg)

    try:
        while True:
            keys.update(T)
            if keys.quit:
                break
            if keys.reset:
                keys.reset = False
                do_reset("[reset] 手动重置")

            # 指令变化率限幅 2 m/s²、10 rad/s²，阶跃指令会让俯仰冲到 50° 直接翻
            vx_t = float(np.clip(keys.vx + trim, -1.6, 1.6))
            vx_cmd += max(-VX_SLEW, min(VX_SLEW, vx_t - vx_cmd))
            wz_cmd += max(-WZ_SLEW, min(WZ_SLEW, keys.wz - wz_cmd))
            cmd.cmd[:] = torch.tensor([[vx_cmd, 0.0, wz_cmd, keys.h]], dtype=torch.float32)

            S = env.state()
            if teacher is not None:
                with torch.no_grad():
                    act = teacher.act(S, cmd.cmd[:1]).clamp(-3.0, 3.0)
            else:
                o = mdp.obs(S, a_prev)
                hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat(
                    (hist[:, 1:], o.unsqueeze(1)), 1)
                with torch.no_grad():
                    act = pi(hist, cmd.goal(torch.tensor([f % 199990]), K)).clamp(-3.0, 3.0)
            env.step(act)
            a_prev = act
            f += 1
            t_alive += T
            n += 1

            # 外环速度修正：策略看不到机身速度（obs 里只有轮速），引擎一换稳态就会偏
            # （实测 h=0.24 时恒定 -0.24 m/s，松手还会漂）。这里用一个慢积分补偿指令。
            v_meas = env.vel_forward
            if a.trim > 0:
                trim = float(np.clip(trim + a.trim * (keys.vx - v_meas) * T, -0.4, 0.4))

            # 终止判定（与 mdp.termination 同口径）
            z, cos_t = env.base_z, env.tilt_cos
            dead = (z < term_h) or (cos_t < TERM_COS) or not np.isfinite(z)
            if dead:
                do_reset(f"[term] 摔了：z={z:.3f} cos={cos_t:.3f}，自动重置")

            wz = float(env.d.qvel[3:6] @ env.d.xmat[env.base_id].reshape(3, 3)[2])
            if n % max(1, a.log_every) == 0:
                print(f"  t={t_alive:5.2f}s z={z:.3f} cos={cos_t:.3f} | "
                      f"cmd {fmt_cmd(keys)} | v={v_meas:+.2f}/{keys.vx:+.2f} "
                      f"w={wz:+.2f}/{keys.wz:+.2f} trim={trim:+.2f}")

            if viewer is not None:
                try:  # 3.x 有 set_texts，老版本没有就忽略
                    viewer.set_texts((f"z={z:.3f}  v={v_meas:+.2f}/{keys.vx:+.2f}  "
                                      f"wz={wz:+.2f}/{keys.wz:+.2f}  h={keys.h:.2f}",))
                except Exception:  # noqa: BLE001
                    pass
                viewer.sync()

            # 实时节拍（严格 50 Hz，落后就全速追）
            next_t += T
            dt = next_t - time.perf_counter()
            if dt > 0:
                time.sleep(dt)

            if a.steps and n >= a.steps:
                break
    except KeyboardInterrupt:
        print("\n[exit] Ctrl+C")
    finally:
        keys.stop()
        if viewer is not None:
            try:
                viewer.close()
            except Exception:  # noqa: BLE001
                pass

    wall = time.perf_counter() - t0
    print(f"\n[done] {n} 控制步 / {t_alive:.2f}s 仿真 / {wall:.1f}s 墙钟 "
          f"({n * T / max(wall, 1e-6):.1f}x 实时)  重置 {n_reset} 次")
    return 0


if __name__ == "__main__":
    sys.exit(main())
