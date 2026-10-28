# SPDX-License-Identifier: MIT
"""从 USD 资产中提取运动学树，用于纯 torch 的可微前向运动学（FK）。"""

from __future__ import annotations

import json
import os

_DEFAULT_NAME = "wheelbipe_v14_2"

_AXIS_VEC = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}


def _vec3(x) -> list[float]:
    return [float(x[0]), float(x[1]), float(x[2])]


def _quat_wxyz(q) -> list[float]:
    """Gf.Quatf -> [w, x, y, z]；也兼容直接给 4 个分量的情况。"""
    if q is None:
        return [1.0, 0.0, 0.0, 0.0]
    if hasattr(q, "GetReal"):  # Gf.Quatf / Quatd
        im = q.GetImaginary()
        return [float(q.GetReal()), float(im[0]), float(im[1]), float(im[2])]
    try:  # Gf.Vec4f，按 wxyz 解释
        return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
    except Exception:  # noqa: BLE001
        return [1.0, 0.0, 0.0, 0.0]


def _parse_axis(a) -> list[float]:
    """``physics:axis`` 通常是 'X'/'Y'/'Z' token，也可能被导出成向量。"""
    if a is None:
        return [0.0, 0.0, 1.0]
    s = str(a).strip().upper()
    if s in _AXIS_VEC:
        return list(_AXIS_VEC[s])
    try:
        v = [float(a[0]), float(a[1]), float(a[2])]
        n = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
        if n > 1e-8:
            return [v[0] / n, v[1] / n, v[2] / n]
    except Exception:  # noqa: BLE001
        pass
    return [0.0, 0.0, 1.0]


def extract_model(usd_path: str, root_link: str | None = None, verbose: bool = True) -> dict:
    """遍历 USD 的物理关节，构造 ``{root, bodies:[...]}`` 形式的运动学树。"""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(os.path.normpath(usd_path))
    if stage is None:
        raise RuntimeError(f"无法打开 USD: {usd_path}")

    rigid = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    rigid_paths = {p.GetPath() for p in rigid}

    joints = []
    for prim in stage.Traverse():
        is_rev = prim.IsA(UsdPhysics.RevoluteJoint)
        is_pri = prim.IsA(UsdPhysics.PrismaticJoint)
        is_fix = prim.IsA(UsdPhysics.FixedJoint)
        if not (is_rev or is_pri or is_fix):
            continue
        jp = UsdPhysics.Joint(prim)
        t0 = jp.GetBody0Rel().GetTargets()
        t1 = jp.GetBody1Rel().GetTargets()
        if not t1:
            continue
        p1 = t1[0]
        p0 = t0[0] if t0 else None
        name = prim.GetName()
        kind = "revolute" if is_rev else ("prismatic" if is_pri else "fixed")
        axis_tok = prim.GetAttribute("physics:axis").Get()
        axis = _parse_axis(axis_tok)
        lo = prim.GetAttribute("physics:lowerLimit").Get()
        hi = prim.GetAttribute("physics:upperLimit").Get()
        loc0 = prim.GetAttribute("physics:localPos0").Get()
        rot0 = prim.GetAttribute("physics:localRot0").Get()
        loc1 = prim.GetAttribute("physics:localPos1").Get()
        rot1 = prim.GetAttribute("physics:localRot1").Get()
        joints.append(
            dict(
                name=name,
                kind=kind,
                axis=list(axis),
                parent_path=str(p0) if p0 is not None else "",
                child_path=str(p1),
                p_J=_vec3(loc0) if loc0 is not None else [0.0, 0.0, 0.0],
                q_J=_quat_wxyz(rot0),
                p_B=_vec3(loc1) if loc1 is not None else [0.0, 0.0, 0.0],
                q_B=_quat_wxyz(rot1),
                lower=None if lo is None else float(lo),
                upper=None if hi is None else float(hi),
            )
        )

    if not joints:
        raise RuntimeError(f"{usd_path} 中没有找到任何 UsdPhysics 关节，无法建树")

    # 只保留 articulation 里真实存在的关节类型（spherical 不进 PhysX articulation）
    joints = [j for j in joints if j["kind"] in ("revolute", "prismatic")]
    if not joints:
        raise RuntimeError(f"{usd_path} 里没有 revolute/prismatic 关节")
    # 注意：physics:body0 / body1 谁是父**并不可靠**（不同导出器写法不同），
    # 所以下面按无向图建树，必要时翻转 localPos0/localPos1。

    def _leaf_name(path: str) -> str:
        return path.rsplit("/", 1)[-1]

    adj = {}
    for j in joints:
        adj.setdefault(j["parent_path"], []).append((j["child_path"], j, False))
        adj.setdefault(j["child_path"], []).append((j["parent_path"], j, True))
    all_paths = list(adj)
    if not all_paths:
        raise RuntimeError(f"{usd_path} 的关节没有 body0/body1 关系")

    want = root_link or "base_link"
    roots = [p for p in all_paths if _leaf_name(p) == want]
    if not roots:  # 退路：度数最小、路径最短的刚体当根
        deg = {p: len(adj[p]) for p in all_paths}
        roots = sorted(all_paths, key=lambda p: (deg[p], p.count("/")))
    root_path = roots[0]

    # BFS（父先于子）
    bodies, index = [], {}
    queue = [(root_path, -1, None, False)]
    while queue:
        path, parent_idx, j, flip = queue.pop(0)
        if path in index:
            continue
        if j is None:
            entry = dict(name=_leaf_name(path), parent=parent_idx, joint=None, kind="root",
                         axis=[0.0, 0.0, 0.0], p_J=[0.0, 0.0, 0.0], q_J=[1.0, 0.0, 0.0, 0.0],
                         p_B=[0.0, 0.0, 0.0], q_B=[1.0, 0.0, 0.0, 0.0], lower=None, upper=None,
                         prim_path=path)  # fmt: skip
        else:
            if flip:
                p_J, q_J, p_B, q_B = j["p_B"], j["q_B"], j["p_J"], j["q_J"]
            else:
                p_J, q_J, p_B, q_B = j["p_J"], j["q_J"], j["p_B"], j["q_B"]
            entry = dict(name=_leaf_name(path), parent=parent_idx, joint=j["name"], kind=j["kind"],
                         axis=j["axis"], p_J=p_J, q_J=q_J, p_B=p_B, q_B=q_B,
                         lower=j["lower"], upper=j["upper"], prim_path=path)  # fmt: skip
        index[path] = len(bodies)
        bodies.append(entry)
        for nxt, jj, fl in adj.get(path, []):
            if nxt not in index:
                queue.append((nxt, index[path], jj, fl))

    # 没被连上的刚体挂到根上（避免后面按名字查不到）；正常情况下不该出现
    orphans = [str(p.GetPath()) for p in rigid if str(p.GetPath()) not in index]
    for p in orphans:
        index[p] = len(bodies)
        bodies.append(dict(name=_leaf_name(p), parent=0, joint=None, kind="fixed",
                           axis=[0.0, 0.0, 0.0], p_J=[0.0, 0.0, 0.0], q_J=[1.0, 0.0, 0.0, 0.0],
                           p_B=[0.0, 0.0, 0.0], q_B=[1.0, 0.0, 0.0, 0.0], lower=None, upper=None,
                           prim_path=p))  # fmt: skip
    if orphans and verbose:
        print(f"[model] 警告：{len(orphans)} 个刚体没有关节连到树上，已按固定件挂在根上: "
              f"{[_leaf_name(p) for p in orphans[:6]]}")

    model = dict(
        usd_path=os.path.abspath(usd_path),
        root=bodies[0]["name"],
        bodies=bodies,
        n_joints_in_tree=sum(1 for b in bodies if b["kind"] in ("revolute", "prismatic")),
    )
    if verbose:
        print(f"[model] {os.path.basename(usd_path)}: {len(bodies)} 刚体, "
              f"{model['n_joints_in_tree']} 可动关节, root={model['root']}")
        for b in bodies:
            print(f"   {b['name']:<26} parent={index_str(bodies, b['parent']):<22} "
                  f"joint={str(b['joint']):<22} {b['kind']}")
    return model


def rigid_paths_str(rigid) -> list[str]:
    return [str(p.GetPath()) for p in rigid]


def index_str(bodies, i):
    return "-" if i < 0 else bodies[i]["name"]


def model_path(name: str = _DEFAULT_NAME) -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", f"{name}_model.json")


def save_model(model: dict, path: str | None = None) -> str:
    path = path or model_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(model, f, indent=1)
    return path


def load_model(path: str | None = None) -> dict:
    path = path or model_path()
    with open(path) as f:
        return json.load(f)


def ensure_model(usd_path: str, path: str | None = None, root_link: str | None = None) -> dict:
    """有 JSON 就读，没有就在当前进程（需要 pxr/Kit）里提取并写盘。"""
    path = path or model_path()
    if os.path.exists(path):
        return load_model(path)
    model = extract_model(usd_path, root_link=root_link)
    save_model(model, path)
    print(f"[model] 已写出 {path}（后续运行不再需要 pxr）")
    return model


def wheel_bodies(model: dict, spec) -> list[str]:
    """从运动学树里找出轮关节对应的子刚体名（left/right_wheel_link）。"""
    out = []
    for b in model["bodies"]:
        jn = b.get("joint") or ""
        if jn.endswith("_wheel_joint"):
            out.append(b["name"])
    return sorted(out)
