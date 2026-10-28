# SPDX-License-Identifier: MIT
"""从 wheelbipe 的 USD 资产生成 MuJoCo MJCF（不启动 Isaac，纯 pxr）。"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

_JOINT_TYPES = ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint", "PhysicsSphericalJoint", "PhysicsFixedJoint")

# 执行器参数
# 来自 robot.py::make_robot_cfg，逐字照抄 Isaac Lab 配置
DM8009_ARMATURE = 1.95e-04 * 9.0 * 9.0
WHEEL_EFFORT = float(os.environ.get("WHEELBIPE_WHEEL_EFFORT", 5.0))
WHEEL_VEL_KD = 0.2

GROUPS = [
    # (正则, 说明, 参数)
    (r".*_(front1|rear1)_joint$", "leg_act",
     dict(act="pos", kp=60.0, kv=2.0, effort=40.0, armature=DM8009_ARMATURE, damp=0.0, vel=17.0)),
    (r".*_(front2|front3|front4|rear2|spring1)_joint$|.*_guide_joint$", "leg_passive",
     dict(act=None, effort=50.0, armature=1e-4, damp=0.01, vel=300.0)),
    (r".*_wheel_joint$", "wheel",
     dict(act="vel", kv=WHEEL_VEL_KD, effort=WHEEL_EFFORT, armature=0.0, damp=0.0, vel=60.0)),
    (r".*_spring2_joint$", "spring",
     dict(act="motor", effort=1000.0, armature=1e-4, damp=50.0, vel=50.0)),
    # Isaac 里 gimbal_yaw 是"零刚度 + 0.5 阻尼"，等于一个自由铰：机器人一动云台自己转
    # （实测 5 s 转 -165°）。它不参与动作/状态，加个轻量伺服锁回 0，观感和实机一致。
    (r"gimbal_yaw_joint$", "gimbal_yaw",
     dict(act="pos", kp=2.0, kv=0.5, effort=2.0, armature=1e-4, damp=0.5, vel=30.0)),
    (r"gimbal_pitch_joint$", "gimbal_pitch",
     dict(act="pos", kp=20.0, kv=0.5, effort=10.0, armature=1e-4, damp=0.5, vel=30.0)),
]


def group_of(name: str):
    import re

    for pat, tag, p in GROUPS:
        if re.fullmatch(pat, name):
            return tag, p
    return None, None


# 基础工具
def _v(x):
    return [float(x[0]), float(x[1]), float(x[2])]


def _f(x, n=6):
    return " ".join(f"{float(v):.{n}g}" for v in x)


def _quat_wxyz(q):
    if hasattr(q, "GetReal"):
        im = q.GetImaginary()
        return [float(q.GetReal()), float(im[0]), float(im[1]), float(im[2])]
    try:
        return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
    except Exception:  # noqa: BLE001
        return [1.0, 0.0, 0.0, 0.0]


def _qnorm(q):
    n = sum(x * x for x in q) ** 0.5
    return [1.0, 0.0, 0.0, 0.0] if n < 1e-12 else [x / n for x in q]


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _qconj(q):
    return [q[0], -q[1], -q[2], -q[3]]


def _qrot(q, v):
    w, x, y, z = q
    t = [2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])]
    return [
        v[0] + w * t[0] + (y * t[2] - z * t[1]),
        v[1] + w * t[1] + (z * t[0] - x * t[2]),
        v[2] + w * t[2] + (x * t[1] - y * t[0]),
    ]


def _axis(a):
    s = str(a).strip().upper()
    m = {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}
    if s in m:
        return m[s]
    try:
        v = [float(a[0]), float(a[1]), float(a[2])]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v] if n > 1e-8 else [0.0, 0.0, 1.0]
    except Exception:  # noqa: BLE001
        return [0.0, 0.0, 1.0]


def _attr(prim, name, default=None):
    a = prim.GetAttribute(name)
    if not a or not a.HasAuthoredValue():
        return default
    v = a.Get()
    return default if v is None else v


def _manual_world(prim):
    """逐级累乘 local transform；对 instance proxy 也正确（XformCache 会错）。"""
    from pxr import Gf, UsdGeom

    chain, cur = [], prim
    while cur is not None:
        chain.append(cur)
        parent = cur.GetParent()
        if parent is None or str(parent.GetPath()) == "/":
            break
        cur = parent
    M = Gf.Matrix4d(1.0)
    for p in reversed(chain):
        x = UsdGeom.Xformable(p)
        if x is not None:
            M = M * x.GetLocalTransformation(0.0)
    return M


# 读 USD
def _collect_geoms(stage, body_paths) -> list:
    from pxr import Gf, Usd, UsdPhysics

    out = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        t = prim.GetTypeName()
        if t not in ("Cube", "Cylinder", "Sphere"):
            continue
        owner, owner_path = None, None
        p = prim.GetParent()
        while p is not None:
            if str(p.GetPath()) in body_paths:
                owner, owner_path = p.GetName(), str(p.GetPath())
                break
            p = p.GetParent()
        if owner is None:
            continue
        M = _manual_world(stage.GetPrimAtPath(owner_path)).GetInverse() * _manual_world(prim)
        tr = Gf.Transform(M)
        g = {
            "type": t,
            "owner": owner,
            "collision": bool(prim.HasAPI(UsdPhysics.CollisionAPI)),
            "pos": _v(tr.GetTranslation()),
            "scale": _v(tr.GetScale()),
            "rot": _quat_wxyz(Gf.Quatf(tr.GetRotation().GetQuat())),
        }
        if t == "Cube":
            g["size"] = float(_attr(prim, "size", 2.0))
        else:
            g["radius"] = float(_attr(prim, "radius", 0.5))
            g["height"] = float(_attr(prim, "height", 1.0))
            g["axis"] = str(_attr(prim, "axis", "Z"))
        out.append(g)
    return out


def dump(usd_path: str) -> dict:
    """读出刚体 / 关节 / 碰撞体，返回纯 dict（可直接 json 序列化）。"""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(os.path.normpath(usd_path))
    if stage is None:
        raise RuntimeError(f"无法打开 USD: {usd_path}")

    bodies, joints = [], []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append({
                "name": prim.GetName(),
                "path": str(prim.GetPath()),
                "mass": float(_attr(prim, "physics:mass", 0.0)),
                "com": _v(_attr(prim, "physics:centerOfMass", (0, 0, 0))),
                "inertia": _v(_attr(prim, "physics:diagonalInertia", (0, 0, 0))),
                "axes": _quat_wxyz(_attr(prim, "physics:principalAxes", (1, 0, 0, 0))),
            })

        tname = prim.GetTypeName()
        if tname not in _JOINT_TYPES:
            continue
        b0 = prim.GetRelationship("physics:body0")
        b1 = prim.GetRelationship("physics:body1")
        j = {
            "name": prim.GetName(),
            "type": tname,
            "body0": str(b0.GetTargets()[0].name) if b0 and b0.GetTargets() else None,
            "body1": str(b1.GetTargets()[0].name) if b1 and b1.GetTargets() else None,
            "pos0": _v(_attr(prim, "physics:localPos0", (0, 0, 0))),
            "rot0": _quat_wxyz(_attr(prim, "physics:localRot0", (1, 0, 0, 0))),
            "pos1": _v(_attr(prim, "physics:localPos1", (0, 0, 0))),
            "rot1": _quat_wxyz(_attr(prim, "physics:localRot1", (1, 0, 0, 0))),
        }
        if tname in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"):
            j["axis"] = _axis(_attr(prim, "physics:axis"))
            j["lower"] = _attr(prim, "physics:lowerLimit")
            j["upper"] = _attr(prim, "physics:upperLimit")
        joints.append(j)

    geoms = _collect_geoms(stage, {b["path"] for b in bodies})
    hist = {}
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        t = prim.GetTypeName() or "-"
        hist[t] = hist.get(t, 0) + 1
    return {"usd": usd_path, "bodies": bodies, "joints": joints, "geoms": geoms, "types": hist}


# 生成 MJCF
def _fmt_limits(j, kind, rnd=5):
    lo, hi = j.get("lower"), j.get("upper")
    if lo is None or hi is None:
        return None
    lo, hi = float(lo), float(hi)
    if abs(lo) > 1e30 and abs(hi) > 1e30:
        return None
    if kind == "hinge":  # USD revolute 限位是度
        lo, hi = lo * 3.141592653589793 / 180.0, hi * 3.141592653589793 / 180.0
    return [round(lo, rnd), round(hi, rnd)]


def build_mjcf(d: dict, *, friction: float = 1.0, guide_collision: bool = False,
               timestep: float = 5e-4, base_h: float = 0.20, solref: str = "0.002 1",
               visual_meshes: dict | None = None) -> str:
    bodies = {b["name"]: b for b in d["bodies"]}
    art = [j for j in d["joints"] if j["type"] in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint")]
    sph = [j for j in d["joints"] if j["type"] == "PhysicsSphericalJoint"]

    # 每个刚体的"父关节"（articulation 树上唯一）
    pj = {j["body1"]: j for j in art}
    root = "base_link"
    if root not in bodies:
        cand = [n for n in bodies if n not in pj]
        root = cand[0]
    children = {}
    for name, j in pj.items():
        children.setdefault(j["body0"], []).append(name)
    for k in children:
        children[k].sort()

    geoms_by_body = {}
    for g in d["geoms"]:
        if g["type"] != "Cube" and g["type"] != "Cylinder":
            continue
        g = dict(g)
        # guide 连杆默认不参与碰撞，但**显示**要保留（不然机器人看着缺零件）
        g["_no_col"] = (not guide_collision) and ("guide" in g["owner"])
        geoms_by_body.setdefault(g["owner"], []).append(g)

    # 球铰 → 在 body1 上加一个无质量锚点刚体（MuJoCo 的 connect 只支持 body1 侧带 anchor）
    anchors = {}
    for k, j in enumerate(sph):
        anchors.setdefault(j["body1"], []).append(
            (f"{j['name']}_anchor", j["pos1"], j["body0"], j["pos0"], j["name"])
        )

    L = []
    A = L.append
    A('<mujoco model="wheelbipe_v14_2">')
    A('  <compiler angle="radian" autolimits="true" eulerseq="xyz"/>')
    A(f'  <option timestep="{timestep}" integrator="implicitfast" solver="Newton" iterations="100"'
      f' tolerance="1e-10" cone="elliptic" impratio="10"/>')
    A('  <visual><global offwidth="1280" offheight="720"/></visual>')
    A('  <default>')
    A(f'    <geom friction="{friction} 0.005 0.0001" contype="2" conaffinity="1" rgba="0.72 0.74 0.78 1"/>')
    # group 3 默认不显示 → 碰撞体藏起来；可见的用 visual / wheel 两个类放在 group 0。
    # 之前只写了 class="collision"，结果窗口里"只有地面没有机器人"（用户实际反馈）。
    A('    <default class="collision"><geom group="3" contype="2" conaffinity="1"/></default>')
    A('    <default class="visual"><geom group="0" contype="0" conaffinity="0"'
      ' rgba="0.62 0.66 0.72 1"/></default>')
    A('    <default class="wheel"><geom group="0" contype="0" conaffinity="0"'
      ' rgba="0.07 0.07 0.08 1"/></default>')
    A('    <default class="guide"><geom group="0" contype="0" conaffinity="0"'
      ' rgba="0.85 0.55 0.15 1"/></default>')
    A('  </default>')
    if visual_meshes:
        A('  <asset>')
        for names in visual_meshes.values():
            for nm in names:
                A(f'    <mesh name="{nm}" file="meshes/{nm}.obj"/>')
        A('  </asset>')
    A('  <worldbody>')
    A('    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>')
    A('    <geom name="floor" type="plane" size="20 20 0.05" contype="1" conaffinity="3"'
      ' rgba="0.45 0.45 0.48 1" friction="1 0.005 0.0001"/>')

    # 广度优先写 body
    def emit_body(name, parent_indent):
        ind = "  " * parent_indent
        b = bodies[name]
        j = pj.get(name)
        if j is None:
            pos, quat = [0.0, 0.0, base_h], [1.0, 0.0, 0.0, 0.0]
        else:
            p_J, R_J = j["pos0"], _qnorm(j["rot0"])
            p_B, R_B = j["pos1"], _qnorm(j["rot1"])
            quat = _qnorm(_qmul(R_J, R_B))
            pos = [p_J[i] + _qrot(R_J, p_B)[i] for i in range(3)]
        A(f'{ind}<body name="{name}" pos="{_f(pos)}" quat="{_f(quat,7)}">')
        ind2 = ind + "  "
        if j is None:
            A(f'{ind2}<freejoint name="root"/>')
        else:
            kind = "hinge" if j["type"] == "PhysicsRevoluteJoint" else "slide"
            # 必须传关节名：正则全是 *_joint$，传刚体名会让 armature/damping 全丢，云台自转到 -165°
            tag, cfg = group_of(j["name"])
            R_B = _qnorm(j["rot1"])
            axis_b = _qrot(_qconj(R_B), j["axis"])
            anchor_b = [-v for v in _qrot(_qconj(R_B), j["pos1"])]
            lim = _fmt_limits(j, kind)
            attrs = [
                f'name="{j["name"]}"', f'type="{kind}"', f'axis="{_f(axis_b,8)}"',
                f'pos="{_f(anchor_b,6)}"',
                f'armature="{cfg["armature"]:g}"' if cfg else "",
                f'damping="{cfg["damp"]:g}"' if cfg and cfg.get("damp") else "",
            ]
            if lim is not None:
                attrs.append(f'range="{_f(lim,6)}"')
            else:
                attrs.append('limited="false"')
            A(f'{ind2}<joint ' + " ".join(a for a in attrs if a) + "/>")
        # 惯性
        A(f'{ind2}<inertial pos="{_f(b["com"])}" quat="{_f(_qnorm(b["axes"]),7)}"'
          f' mass="{b["mass"]:.8g}" diaginertia="{_f(b["inertia"],7)}"/>')
        # 碰撞 + 显示几何：同一个形状写两遍（visual 在 group 0 可见且不参与碰撞，
        # collision 在 group 3 隐藏但参与碰撞）
        is_wheel = name.endswith("wheel_link")
        is_guide = "guide" in name
        use_mesh = bool(visual_meshes and name in visual_meshes)   # 有真网格就不用基元当显示
        for gi, g in enumerate(geoms_by_body.get(name, [])):
            if g["type"] == "Cube":
                half = [0.5 * g["size"] * s for s in g["scale"]]
                base = f"{name}_box{gi}"
                shape = (f'type="box" size="{_f(half)}" pos="{_f(g["pos"])}" '
                         f'quat="{_f(_qnorm(g["rot"]),7)}"')
            else:
                r = g["radius"]
                hl = 0.5 * g["height"]
                q = _qnorm(g["rot"])
                ax = g.get("axis", "Z").upper()
                if ax == "X":
                    q = _qnorm(_qmul(q, [0.7071067811865476, 0.0, 0.7071067811865476, 0.0]))
                elif ax == "Y":
                    q = _qnorm(_qmul(q, [0.7071067811865476, -0.7071067811865476, 0.0, 0.0]))
                base = f"{name}_cyl{gi}"
                shape = (f'type="cylinder" size="{r:.6g} {hl:.6g}" pos="{_f(g["pos"])}" '
                         f'quat="{_f(q,7)}"')
            vcls = "wheel" if is_wheel else ("guide" if is_guide else "visual")
            if not use_mesh:
                A(f'{ind2}<geom name="{base}_vis" {shape} class="{vcls}"/>')
            if not g.get("_no_col"):
                A(f'{ind2}<geom name="{base}" {shape} class="collision"/>')
        # 真网格显示层（USD 里 78 个 Mesh，比基元好看得多）
        if use_mesh:
            vcls = "wheel" if is_wheel else ("guide" if is_guide else "visual")
            for nm in visual_meshes[name]:
                A(f'{ind2}<geom name="{nm}_vis" type="mesh" mesh="{nm}" class="{vcls}"/>')
        # 球铰锚点（无质量占位刚体）
        for aname, apos, _b0, _p0, _jn in anchors.get(name, []):
            A(f'{ind2}<body name="{aname}" pos="{_f(apos)}">')
            A(f'{ind2}  <inertial pos="0 0 0" mass="1e-9" diaginertia="1e-12 1e-12 1e-12"/>')
            A(f'{ind2}</body>')
        for c in children.get(name, []):
            emit_body(c, parent_indent + 1)
        A(f'{ind} </body>')

    emit_body(root, 2)
    A('  </worldbody>')

    # 执行器
    A('  <actuator>')
    for j in art:
        name = j["name"]
        tag, cfg = group_of(name)
        if cfg is None or cfg.get("act") is None:
            continue
        if cfg["act"] == "pos":
            lo = -1.5 if tag == "leg_act" else -1.2
            hi = 1.5 if tag == "leg_act" else 0.0
            A(f'    <position name="{name}_act" joint="{name}" kp="{cfg["kp"]:g}" kv="{cfg["kv"]:g}"'
              f' ctrlrange="{lo:g} {hi:g}" forcerange="-{cfg["effort"]:g} {cfg["effort"]:g}"/>')
        elif cfg["act"] == "vel":
            lim = 10.0 * 3.0
            A(f'    <velocity name="{name}_act" joint="{name}" kv="{cfg["kv"]:g}"'
              f' ctrlrange="-{lim:g} {lim:g}" forcerange="-{cfg["effort"]:g} {cfg["effort"]:g}"/>')
        elif cfg["act"] == "motor":
            A(f'    <motor name="{name}_act" joint="{name}" gear="1"'
              f' ctrlrange="-{cfg["effort"]:g} {cfg["effort"]:g}"'
              f' forcerange="-{cfg["effort"]:g} {cfg["effort"]:g}"/>')
    A('  </actuator>')

    # 闭链
    A('  <equality>')
    for j in sph:
        A(f'    <connect name="{j["name"]}" body1="{j["body0"]}" body2="{j["name"]}_anchor"'
          f' anchor="{_f(j["pos0"])}" solref="{solref}" solimp="0.95 0.99 0.001"/>')
    A('  </equality>')
    A('</mujoco>')
    return "\n".join(L) + "\n"


def export_visual_meshes(usd_path: str, mesh_dir: str, body_names: list[str],
                         verbose: bool = True) -> dict[str, list[str]]:
    """从 USD 里把**显示网格**导成 OBJ（MuJoCo 里当 visual geom 用）。"""
    import glob
    import os

    from pxr import Gf, Usd, UsdGeom

    # 网格在**显示层**：wheelbipeV14_2.usd 只引用物理层（Cube/Cylinder 基元），
    # 真网格在 configuration/wheelbipeV14_2_base.usd 的 /visuals/<link>/... 下。
    cands = glob.glob(os.path.join(os.path.dirname(os.path.normpath(usd_path)), "configuration", "*_base.usd"))
    if not cands:
        cands = glob.glob(os.path.join(os.path.dirname(os.path.normpath(usd_path)), "*_base.usd"))
    if not cands:
        if verbose:
            print("[usd2mjcf] 没找到 *_base.usd（显示层），跳过网格导出")
        return {}
    vis_usd = cands[0]

    stage = Usd.Stage.Open(os.path.normpath(vis_usd), load=Usd.Stage.LoadAll)
    if stage is None:
        return {}
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    names = set(body_names)

    os.makedirs(mesh_dir, exist_ok=True)
    out: dict[str, list[str]] = {}
    n_skip = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        # 只要**显示层**（/visuals）：这个 USD 里 /meshes 是同一批网格的第二份，
        # /colliders 是碰撞网格。全都导进来 -> 底盘/云台看着"叠了两层"。
        parts = prim.GetPath().pathString.split("/")
        if len(parts) < 3 or parts[1].lower() != "visuals":
            n_skip += 1
            continue
        # 参考系取最外层 /visuals/<link>，内层丢了 STL 校正旋转，机器人歪 35°
        owner, q, frame = None, prim.GetPath(), None
        while q and str(q) != "/":
            if q.name in names:
                owner, frame = q.name, q
            q = q.GetParentPath()
        if owner is None:
            n_skip += 1
            continue
        m = UsdGeom.Mesh(prim)
        pts = m.GetPointsAttr().Get()
        idx = m.GetFaceVertexIndicesAttr().Get()
        cnt = m.GetFaceVertexCountsAttr().Get()
        if not pts or not idx or not cnt:
            n_skip += 1
            continue
        rel = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(frame)).GetInverse() * \
            cache.GetLocalToWorldTransform(prim)
        name = f"{owner}_{len(out.get(owner, []))}"
        path = os.path.join(mesh_dir, name + ".obj")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {prim.GetPath()} -> {owner}\n")
            for p in pts:
                w = rel.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
                f.write(f"v {w[0]:.6f} {w[1]:.6f} {w[2]:.6f}\n")
            k = 0
            for c in cnt:
                c = int(c)
                if c == 3:
                    f.write(f"f {idx[k] + 1} {idx[k + 1] + 1} {idx[k + 2] + 1}\n")
                elif c == 4:
                    f.write(f"f {idx[k] + 1} {idx[k + 1] + 1} {idx[k + 3] + 1} {idx[k + 2] + 1}\n")
                else:  # 扇形三角化
                    for j in range(1, c - 1):
                        f.write(f"f {idx[k] + 1} {idx[k + j] + 1} {idx[k + j + 1] + 1}\n")
                k += c
        out.setdefault(owner, []).append(name)
    if verbose:
        print(f"[usd2mjcf] 显示网格: {sum(len(v) for v in out.values())} 个 / "
              f"{len(out)} 个刚体（跳过 {n_skip} 个不在刚体下的 Mesh）")
    return out


def find_usd() -> str:
    """本地找 USD（不能 import robot.py：它 import isaaclab，CPU 机器上没有）。"""
    env = os.environ.get("WHEELBIPE_USD")
    if env:
        if not os.path.exists(env):
            raise FileNotFoundError(f"WHEELBIPE_USD 指向的文件不存在: {env}")
        return env
    root = os.environ.get("WHEELBIPE_ASSET_ROOT")
    roots = [root] if root else []
    roots.append(os.path.join(REPO, "..", "wheeled-legged_RL", "source", "agent_world",
                              "agent_world", "assets", "usd_files"))
    roots.append(os.path.join(REPO, "assets", "usd_files"))
    for r in roots:
        if not r or not os.path.isdir(r):
            continue
        for name in ("wheelbipeV14_2.usd", "wheelbipeV14_2_ng.usd"):
            p = os.path.normpath(os.path.join(r, "wheelbipeV14_2_1", name))
            if os.path.exists(p):
                return p
    raise FileNotFoundError("找不到 wheelbipe USD，请设 WHEELBIPE_ASSET_ROOT 或 --usd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", type=str, default=None)
    ap.add_argument("--dump", action="store_true", help="只打印拓扑")
    ap.add_argument("--json", type=str, default=None, help="把解析结果写成 json")
    ap.add_argument("--fk-json", action="store_true", help="同时生成 FK 用的 data/*_model.json")
    ap.add_argument("--guide-collision", action="store_true",
                    help="打开 16 个 guide 连杆的碰撞（默认关：USD 里 left_bottom1 等在 y=-0.146，"
                         "左右不对称，被地面剐住会让策略只能左转不能右转）")
    ap.add_argument("--friction", type=float, default=1.0)
    ap.add_argument("--timestep", type=float, default=5e-4)
    ap.add_argument("--no-meshes", action="store_true",
                    help="不导出显示网格（默认导：物理层只有 Cube/Cylinder 基元，看着像个箱子）")
    ap.add_argument("--out", type=str, default=os.path.join(REPO, "assets", "mujoco", "wheelbipe_v14_2.xml"))
    a = ap.parse_args()

    usd = a.usd or find_usd()
    print(f"[usd2mjcf] USD = {usd}")
    d = dump(usd)
    mass = sum(b["mass"] for b in d["bodies"])
    print(f"[usd2mjcf] bodies={len(d['bodies'])} joints={len(d['joints'])} "
          f"geoms={len(d['geoms'])} total_mass={mass:.3f} kg")

    if a.dump:
        for j in d["joints"]:
            lim = _fmt_limits(j, "hinge" if "Revolute" in j["type"] else "slide")
            print(f"  {j['name']:<28s} {j['type'][7:]:<10s} {str(j['body0']):<24s} -> {str(j['body1']):<24s}"
                  f" axis={j.get('axis')} lim={lim}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        print(f"[usd2mjcf] 写出 {a.json}")

    vm = None
    if not a.no_meshes:
        mesh_dir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "meshes")
        vm = export_visual_meshes(usd, mesh_dir, [b["name"] for b in d["bodies"]])

    xml = build_mjcf(d, friction=a.friction, guide_collision=a.guide_collision,
                     timestep=a.timestep, visual_meshes=vm)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"[usd2mjcf] 写出 {a.out}  ({len(xml)} 字节)")

    if a.fk_json:
        from supermimic_wheelbipe.model import extract_model, model_path, save_model

        m = extract_model(usd, verbose=False)
        p = save_model(m, model_path())
        print(f"[usd2mjcf] FK 模型 {p}  ({len(m['bodies'])} 刚体)")


if __name__ == "__main__":
    main()
