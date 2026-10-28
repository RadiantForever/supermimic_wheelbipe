# SPDX-License-Identifier: MIT
"""手柄（Windows XInput，纯 ctypes 零依赖）+ 手柄→速度指令映射。"""

from __future__ import annotations

import ctypes
import math
import sys
from ctypes import wintypes

BUTTONS = {
    "DPAD_UP": 0x0001, "DPAD_DOWN": 0x0002, "DPAD_LEFT": 0x0004, "DPAD_RIGHT": 0x0008,
    "START": 0x0010, "BACK": 0x0020, "L3": 0x0040, "R3": 0x0080,
    "LB": 0x0100, "RB": 0x0200, "A": 0x1000, "B": 0x2000, "X": 0x4000, "Y": 0x8000,
}


class _GAMEPAD(ctypes.Structure):
    _fields_ = [
        ("wButtons", wintypes.WORD),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]


class _STATE(ctypes.Structure):
    _fields_ = [("dwPacketNumber", wintypes.DWORD), ("Gamepad", _GAMEPAD)]


def _dz(v: float, dz: float) -> float:
    """死区 + 重新归一化：出死区从 0 开始，到满杆是 ±1。"""
    a = abs(v)
    if a <= dz:
        return 0.0
    return math.copysign(min((a - dz) / max(1e-6, 1.0 - dz), 1.0), v)


class XInputPad:
    """只读、非阻塞。``poll()`` 返回 None 表示没插手柄 / 被拔了。"""

    def __init__(self, index: int = 0):
        self.index = int(index)
        self.ok = False
        self.err = ""
        self._lib = None
        if sys.platform != "win32":
            self.err = "XInput 只在 Windows 可用"
            return
        for name in ("xinput1_4", "xinput1_3", "xinput9_1_0"):
            try:
                self._lib = ctypes.windll.LoadLibrary(name)
                break
            except OSError:
                continue
        if self._lib is None:
            self.err = "找不到 xinput*.dll（系统太旧？）"
            return
        try:
            self._get = self._lib.XInputGetState
            self._get.argtypes = [wintypes.DWORD, ctypes.POINTER(_STATE)]
            self._get.restype = wintypes.DWORD
        except AttributeError:
            self.err = "xinput 库缺少 XInputGetState"
            return
        self.ok = True

    def poll(self):
        if not self.ok:
            return None
        st = _STATE()
        if self._get(self.index, ctypes.byref(st)) != 0:  # 1167 = ERROR_DEVICE_NOT_CONNECTED
            return None
        g = st.Gamepad
        return {
            "lx": max(-1.0, g.sThumbLX / 32768.0), "ly": max(-1.0, g.sThumbLY / 32768.0),
            "rx": max(-1.0, g.sThumbRX / 32768.0), "ry": max(-1.0, g.sThumbRY / 32768.0),
            "lt": g.bLeftTrigger / 255.0, "rt": g.bRightTrigger / 255.0,
            "buttons": int(g.wButtons),
        }


def _pressed(state, name) -> bool:
    return bool(state and (state["buttons"] & BUTTONS[name]))


class GamepadCmd:
    """把键盘状态（``KeyboardCmd``）和手柄合成一份指令，接口对调用方完全不变。"""

    def __init__(self, kb, pad: XInputPad | None = None, *, enabled: bool = True,
                 vx_max: float = 1.5, wz_max: float = 10.21, h_rate: float = 0.12,
                 h_lo: float = 0.16, h_hi: float = 0.28, deadzone: float = 0.08,
                 expo: float = 0.0, wz_ramp: float = 0.0, h_trigger_rate: float = 0.15,
                 h_presets=(0.18, 0.20, 0.22, 0.24)):
        self.kb = kb
        self.pad = pad
        self.enabled = bool(enabled and pad is not None and pad.ok)
        self.vx_max, self.wz_max, self.h_rate = float(vx_max), float(wz_max), float(h_rate)
        self.h_lo, self.h_hi = float(h_lo), float(h_hi)
        self.h_trigger_rate = float(h_trigger_rate)
        self.h_presets = tuple(float(v) for v in h_presets)
        self._preset_i = 0
        self.dz, self.expo, self.wz_ramp = float(deadzone), float(expo), float(wz_ramp)
        self._latched = False
        self._recentered = True
        self._prev_btn = 0
        self._wz = 0.0
        # 构造时先探一次，方便调用方在启动横幅里就显示"手柄已接管"
        self.connected = bool(self.enabled and pad.poll() is not None)

    # 内部
    def _shape(self, v: float) -> float:
        s = _dz(v, self.dz)
        if self.expo > 0:
            s = math.copysign(abs(s) ** (1.0 + self.expo), s)
        return s

    # 对外
    def update(self, dt: float = 0.02):
        self.kb.update()
        if not self.enabled:
            return
        st = self.pad.poll()
        self.connected = st is not None
        if st is None:
            self._latched = False
            return

        btn, prev = st["buttons"], self._prev_btn
        self._prev_btn = btn
        edge = lambda name: bool(btn & BUTTONS[name]) and not bool(prev & BUTTONS[name])  # noqa: E731

        # 急停锁：按 A 锁死；**必须先松杆回中、再推**才解锁
        # （否则杆已经推着的时候按 A，会在同一帧被"杆偏离"立刻解锁，急停失效）
        if edge("A"):
            self._latched = True
            self._recentered = False
        s_ly, s_rx, s_ry = self._shape(st["ly"]), self._shape(st["rx"]), self._shape(st["ry"])
        if self._latched:
            if abs(s_ly) < 0.05 and abs(s_rx) < 0.05:
                self._recentered = True
            elif self._recentered:
                self._latched = False

        if self._latched:
            self.kb.vx, self.kb.wz = 0.0, 0.0
            self._wz = 0.0
        else:
            self.kb.vx = s_ly * self.vx_max
            tgt = s_rx * self.wz_max
            if self.wz_ramp > 0:
                step = self.wz_ramp * dt
                self._wz += max(-step, min(step, tgt - self._wz))
            else:
                self._wz = tgt
            self.kb.wz = self._wz

        # 抬腿 / 变高（这台车两条腿联动，只能一起升降，不能单独抬一条）
        # 四种方式，按显眼程度排序：X 一键循环预设 > 十字键 ±0.02 > 扳机连续 > 右杆上下连续
        if edge("X"):
            self.kb.h = self.h_presets[self._preset_i % len(self.h_presets)]
            self._preset_i += 1
            self.kb.h = min(max(self.kb.h, self.h_lo), self.h_hi)
        else:
            dh = s_ry * self.h_rate * dt
            lt = st["lt"] if st["lt"] > 0.10 else 0.0
            rt = st["rt"] if st["rt"] > 0.10 else 0.0
            dh += (rt - lt) * self.h_trigger_rate * dt
            if edge("LB") or edge("DPAD_DOWN"):
                dh -= 0.02
            if edge("RB") or edge("DPAD_UP"):
                dh += 0.02
            self.kb.h = min(max(self.kb.h + dh, self.h_lo), self.h_hi)

        if edge("B"):
            self.kb.reset = True
        if edge("START"):
            self.kb.quit = True

    def stop(self):
        self.kb.stop()

    # 让调用方照旧用 keys.vx / keys.wz / keys.h / keys.reset / keys.quit
    # （显式 property 而不是 __getattr__：调用方会写 keys.reset = False，转发必须双向）
    @property
    def vx(self):
        return self.kb.vx

    @vx.setter
    def vx(self, v):
        self.kb.vx = v

    @property
    def wz(self):
        return self.kb.wz

    @wz.setter
    def wz(self, v):
        self.kb.wz = v

    @property
    def h(self):
        return self.kb.h

    @h.setter
    def h(self, v):
        self.kb.h = v

    @property
    def reset(self):
        return self.kb.reset

    @reset.setter
    def reset(self, v):
        self.kb.reset = v

    @property
    def quit(self):
        return self.kb.quit

    @quit.setter
    def quit(self, v):
        self.kb.quit = v


def describe(pad: XInputPad | None) -> str:
    if pad is None:
        return "手柄：未启用"
    if not pad.ok:
        return f"手柄：不可用（{pad.err}）"
    return "手柄：XInput 已就绪" if pad.poll() else "手柄：XInput 就绪，但当前没检测到设备"


def selftest(pad: XInputPad | None = None, n: int = 5):
    """没有手柄也能验证映射曲线：喂合成杆值，打印输出的指令。"""
    class _Fake:
        ok = True

        def __init__(self):
            self.state = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0, "lt": 0.0, "rt": 0.0,
                          "buttons": 0}

        def poll(self):
            return self.state

    class _KB:
        def __init__(self):
            self.vx = self.wz = 0.0
            self.h = 0.20
            self.reset = self.quit = False

        def update(self):
            pass

        def stop(self):
            pass

    fake = _Fake()
    cmd = GamepadCmd(_KB(), fake, vx_max=1.5, wz_max=10.21, deadzone=0.08)
    print("  杆值 -> 指令（左杆上下->vx，右杆左右->wz，右杆上下->h 速率）")
    print(f"  {'ly':>6s} {'rx':>6s} {'ry':>6s}      {'vx':>6s} {'wz':>7s} {'h':>6s}")
    for ly, rx, ry in [(0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.1, 0.0, 0.0),
                       (0.3, 0.0, 0.0), (0.6, 0.0, 0.0), (1.0, 0.0, 0.0),
                       (-1.0, 0.0, 0.0), (0.6, 0.3, 0.0), (0.6, 1.0, 0.0),
                       (0.6, -1.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, -0.5)]:
        cmd.kb.h = 0.20
        fake.state.update(ly=ly, rx=rx, ry=ry)
        cmd.update(dt=0.02)
        print(f"  {ly:+6.2f} {rx:+6.2f} {ry:+6.2f}   -> {cmd.kb.vx:+6.3f} {cmd.kb.wz:+7.3f} "
              f"{cmd.kb.h:6.3f}")
    print("\n  说明：ry 每次只作用 0.02 s，所以 h 只变 0.12*0.02*ry；连续推就是连续变高。")


__all__ = ["XInputPad", "GamepadCmd", "describe", "selftest", "BUTTONS"]
