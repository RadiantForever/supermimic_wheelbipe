# SPDX-License-Identifier: MIT
"""wheelbipe V14_2 机器人：资产定位、ArticulationCfg、关节分组（RobotSpec）、动作尺度、弹簧力模型。"""

from __future__ import annotations

import os
import re

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .constants import (  # noqa: F401  (re-export，其他模块从这里拿常量)
    ACTION_CLIP,
    ACTION_SCALE_LEG,
    ACTION_SCALE_WHEEL,
    CONTROL_DT,
    DECIMATION,
    SIM_DT,
    SPRING_BIAS_RANGE,
    SPRING_FORCE_DOWN,
    SPRING_FORCE_UP,
    SPRING_LENGTH,
    SPRING_OFFSET,
    WHEEL_RADIUS,
    WHEEL_EFFORT,
    WHEEL_VEL_KD,
    WHEEL_VEL_LIMIT,
    spring_force,
)

# 资产
# 可以用环境变量覆盖；否则按常见目录顺序查找
_ASSET_SUBDIR = os.path.join("source", "agent_world", "agent_world", "assets", "usd_files")


def _candidates() -> list[str]:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = []
    env = os.environ.get("WHEELBIPE_ASSET_ROOT")
    if env:
        out.append(env)
    out += [
        os.path.join(here, "..", "wheeled-legged_RL", _ASSET_SUBDIR),
        os.path.join(here, "assets"),
    ]
    try:  # 如果 wheeled-legged_RL 已经 pip install -e，直接用它的 AssetPath
        import agent_world  # noqa: F401

        out.insert(0, os.path.join(str(agent_world.AssetPath), "usd_files"))
    except Exception:
        pass
    return [os.path.normpath(p) for p in out]


def asset_dir() -> str:
    for p in _candidates():
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        "找不到 wheelbipe 的 USD 资产目录。请设置环境变量 WHEELBIPE_ASSET_ROOT 指向 "
        "wheeled-legged_RL/source/agent_world/agent_world/assets/usd_files"
    )


def usd_path(no_gimbal: bool = False) -> str:
    """无云台版优先（云台不需要，少 2 个 DOF 也更省事）。"""
    forced = os.environ.get("WHEELBIPE_USD")
    if forced:
        if not os.path.exists(forced):
            raise FileNotFoundError(f"WHEELBIPE_USD 指向的文件不存在: {forced}")
        return forced
    d = asset_dir()
    order = ["wheelbipeV14_2_1/wheelbipeV14_2_ng.usd", "wheelbipeV14_2_1/wheelbipeV14_2.usd"]
    if not no_gimbal:
        order = order[::-1]
    for rel in order:
        p = os.path.join(d, rel.replace("/", os.sep))
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{d} 下找不到 wheelbipeV14_2 的 USD")


# 执行器/物理参数
DM8009_ARMATURE = 1.95e-04 * 9.0 * 9.0  # 与 wheeled-legged_RL 保持一致


# ArticulationCfg
def _usd_joint_names(usd: str) -> list[str]:
    """从 USD 里读出所有 physics joint 的名字，用来过滤执行器正则。"""
    try:
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(usd)
        names = []
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                names.append(prim.GetName())
        return names
    except Exception as e:  # noqa: BLE001
        print(f"[robot] 读 USD 关节名失败（{e}），执行器按完整正则配置")
        return []


def _act(patterns, names, cls=IdealPDActuatorCfg, effort=None, velocity=None, **kw):
    """按真实关节名裁剪正则；整组都不匹配则返回 None（该执行器不建）。"""
    pats = list(patterns)
    if names:
        pats = [p for p in pats if any(re.fullmatch(p, n) for n in names)]
        if not pats:
            return None
        for k, v in list(kw.items()):
            if isinstance(v, dict):
                kw[k] = {pk: pv for pk, pv in v.items() if any(re.fullmatch(pk, n) for n in names)}
    if effort is not None:
        kw["effort_limit"] = effort
        kw["effort_limit_sim"] = effort
    if velocity is not None:
        kw["velocity_limit"] = velocity
        kw["velocity_limit_sim"] = velocity
    return cls(joint_names_expr=pats, **kw)


def make_robot_cfg(usd: str | None = None, prim_path: str = "{ENV_REGEX_NS}/Robot") -> ArticulationCfg:
    """执行器/物理参数逐项照抄 ``wheeled-legged_RL/.../wheelbipe_V14_2.py``。"""
    usd = usd or usd_path()
    names = _usd_joint_names(usd)
    groups = [
        # 4 个主动腿关节（位置伺服，对应动作 a[:4]）
        (
            "legs_act",
            _act(
                [".*_rear1_joint", ".*_front1_joint"], names,
                stiffness={".*_rear1_joint": 60.0, ".*_front1_joint": 60.0},
                damping={".*_rear1_joint": 2.0, ".*_front1_joint": 2.0},
                effort=40.0, velocity=17.0, armature=DM8009_ARMATURE,
            ),
        ),
        # 被动连杆 + 直线导轨：只留极小阻尼
        (
            "legs_inact",
            _act(
                [".*_rear2_joint", ".*_front2_joint", ".*_front3_joint", ".*_front4_joint",
                 ".*_spring1_joint", ".*_guide_joint"], names,
                stiffness=0.0, damping=0.01, effort=50.0, velocity=300.0, armature=1e-4,
            ),
        ),
        # 轮速伺服必须用 ImplicitActuatorCfg，IdealPD 保持 20 ms 力矩会让低惯量轮子振荡到 ±50 rad/s
        (
            "wheel",
            _act([".*_wheel_joint"], names, damping=WHEEL_VEL_KD,
                 stiffness=0.0, effort=WHEEL_EFFORT, velocity=60.0, armature=0.0),
        ),
        # 弹簧滑块：每个控制步用 set_joint_effort_target 打力，所以这里只给阻尼
        (
            "spring",
            _act([".*_spring2_joint"], names,
                 stiffness=0.0, damping=50.0, effort=1000.0, velocity=50.0, armature=1e-4),
        ),
        # 云台（只有带云台的 USD 才有；不进动作/状态，只做伺服+阻尼）
        (
            "gimbal_yaw",
            _act(["gimbal_yaw_joint"], names,
                 stiffness=0.0, damping=0.5, effort=2.0, velocity=30.0, armature=1e-4),
        ),
        (
            "gimbal_pitch",
            _act(["gimbal_pitch_joint"], names,
                 stiffness=20.0, damping=0.5, effort=10.0, velocity=30.0, armature=1e-4),
        ),
    ]
    actuators = {k: v for k, v in groups if v is not None}
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd,
            activate_contact_sensors=False,
            copy_from_source=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=False,
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=6,
            ),
        ),
        # 显式把初始关节角写成 0：默认姿态 = 腿全伸直的姿态，重置高度由 FK 估计
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.38), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}
        ),
        actuators=actuators,
    )


# 关节分组
# 实现在 spec.py（不依赖 isaaclab），这里只做 re-export
from .spec import RobotSpec, build_spec  # noqa: E402,F401
