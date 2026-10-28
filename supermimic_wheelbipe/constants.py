# SPDX-License-Identifier: MIT
"""不依赖 isaaclab 的常量，供 CPU 自测使用。"""

from __future__ import annotations

import torch

# 时间
DECIMATION = 4
SIM_DT = 1.0 / 200.0
CONTROL_DT = SIM_DT * DECIMATION  # 50 Hz

# 轮半径
WHEEL_RADIUS = 0.06

# 动作：与上游 leg/wheel action_scale 一致
ACTION_SCALE_LEG = 0.5      # rad / 单位动作
ACTION_SCALE_WHEEL = 10.0   # rad/s / 单位动作
ACTION_CLIP = 3.0
WHEEL_VEL_LIMIT = 150.0     # max_wheel_vel(100) * 1.5

# 轮速伺服阻尼，配合 effort_limit 5 N·m；取 5.0 时 1 rad/s 误差即满力矩，0.2 带载跟不上
WHEEL_VEL_KD = 0.2

# 轮力矩上限，USD 写 5.0、上游 env.yaml 写 20.0，可用环境变量覆盖
WHEEL_EFFORT = float(__import__("os").environ.get("WHEELBIPE_WHEEL_EFFORT", 5.0))

# 弹簧，对应上游 spring_settings(linear)
SPRING_OFFSET = 0.06076
SPRING_FORCE_DOWN = 400.0
SPRING_FORCE_UP = 600.0
SPRING_LENGTH = 0.07
SPRING_BIAS_RANGE = 50.0


def spring_force(q_spring: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """线性弹簧力，压缩越多越大；bias 每次 reset 重采样。"""
    length = torch.clamp(SPRING_OFFSET - q_spring, min=0.0)
    return SPRING_FORCE_DOWN + (SPRING_FORCE_UP - SPRING_FORCE_DOWN) / SPRING_LENGTH * length + bias


def height_window(z_stand: float) -> tuple[float, float]:
    """高度指令窗口 (lo, hi)，绝对根高；取稳态高度上方 0.01~0.09 m，夹在 [0.17, 0.34]。"""
    lo = min(max(z_stand + 0.01, 0.17), 0.26)
    hi = min(max(z_stand + 0.09, 0.22), 0.34)
    return lo, max(hi, lo + 0.04)


def default_height_cmd(z_stand: float) -> float:
    lo, hi = height_window(z_stand)
    return lo + 0.5 * (hi - lo)


def default_term_height(z_stand: float) -> float:
    """低于稳态高度 6 cm 视为摔倒。"""
    return max(z_stand - 0.06, 0.08)
