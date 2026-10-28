# SPDX-License-Identifier: MIT
"""关节分组（RobotSpec）：把 articulation 的关节名映射成网络/世界模型需要的维度。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PAT_WHEEL = re.compile(r".*_wheel_joint$")
_PAT_LEG_ACT = re.compile(r".*_(front1|rear1)_joint$")
_PAT_LEG_ALL = re.compile(r".*_(front1|front2|front3|front4|rear1|rear2)_joint$")
_PAT_SPRING = re.compile(r".*_spring2_joint$")
_PAT_GIMBAL = re.compile(r"gimbal_.*")


@dataclass
class RobotSpec:
    joint_names: list[str]
    q_idx: list[int]          # 进状态的"位置"维度（去掉轮子和云台）
    qd_idx: list[int]         # 进状态的"速度"维度（去掉云台）
    act_leg_idx: list[int]    # 动作：腿（位置目标）
    act_wheel_idx: list[int]  # 动作：轮（速度目标）
    wheel_idx: list[int]
    spring_idx: list[int]     # 需要施加弹簧力的关节
    leg_all_idx: list[int]    # 全部腿关节（含被动）
    gimbal_idx: list[int]
    q_lo: object = None       # torch tensor，长度 = len(joint_names)
    q_hi: object = None
    extra: dict = field(default_factory=dict)

    @property
    def act_idx(self) -> list[int]:
        return self.act_leg_idx + self.act_wheel_idx

    @property
    def nq(self) -> int:
        return len(self.q_idx)

    @property
    def nd(self) -> int:
        return len(self.qd_idx)

    @property
    def na(self) -> int:
        return len(self.act_leg_idx) + len(self.act_wheel_idx)

    @property
    def nj(self) -> int:
        return len(self.joint_names)

    def describe(self) -> str:
        show = lambda ids: [self.joint_names[i] for i in ids]  # noqa: E731
        return (
            f"joints={self.nj} (leg_act={len(self.act_leg_idx)} wheel={len(self.act_wheel_idx)} "
            f"spring={len(self.spring_idx)} gimbal={len(self.gimbal_idx)})\n"
            f"  act_leg   : {show(self.act_leg_idx)}\n"
            f"  act_wheel : {show(self.act_wheel_idx)}\n"
            f"  q  (state): {show(self.q_idx)}\n"
            f"  qd (state): {show(self.qd_idx)}\n"
            f"  spring    : {show(self.spring_idx)}\n"
            f"  gimbal    : {show(self.gimbal_idx)}"
        )


def build_spec(joint_names, q_lo=None, q_hi=None) -> RobotSpec:
    names = list(joint_names)
    wheel = [i for i, n in enumerate(names) if _PAT_WHEEL.match(n)]
    leg_act = [i for i, n in enumerate(names) if _PAT_LEG_ACT.match(n) and not _PAT_WHEEL.match(n)]
    leg_all = [i for i, n in enumerate(names) if _PAT_LEG_ALL.match(n)]
    spring = [i for i, n in enumerate(names) if _PAT_SPRING.match(n)]
    gimbal = [i for i, n in enumerate(names) if _PAT_GIMBAL.match(n)]
    if len(leg_act) != 4:
        raise RuntimeError(
            f"期望 4 个主动腿关节（*_front1_joint / *_rear1_joint），实际 {len(leg_act)}: "
            f"{[names[i] for i in leg_act]}\n全部关节: {names}"
        )
    if len(wheel) != 2:
        raise RuntimeError(f"期望 2 个轮关节（*_wheel_joint），实际 {len(wheel)}\n全部关节: {names}")
    qd_idx = [i for i in range(len(names)) if i not in set(gimbal)]
    q_idx = [i for i in qd_idx if i not in set(wheel)]
    return RobotSpec(
        joint_names=names,
        q_idx=q_idx,
        qd_idx=qd_idx,
        act_leg_idx=leg_act,
        act_wheel_idx=wheel,
        wheel_idx=wheel,
        spring_idx=spring,
        leg_all_idx=leg_all,
        gimbal_idx=gimbal,
        q_lo=q_lo,
        q_hi=q_hi,
    )
