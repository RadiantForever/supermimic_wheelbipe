# SPDX-License-Identifier: MIT
"""从 wheelbipe 的 USD 提取运动学树，写成 data/wheelbipe_v14_2_model.json。"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supermimic_wheelbipe.model import extract_model, model_path, save_model  # noqa: E402
from supermimic_wheelbipe.robot import usd_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", type=str, default=None)
    ap.add_argument("--out", type=str, default=model_path())
    ap.add_argument("--root", type=str, default=None, help="根连杆名（默认自动找）")
    a = ap.parse_args()
    usd = a.usd or usd_path()
    print(f"[extract] USD = {usd}")
    model = extract_model(usd, root_link=a.root)
    path = save_model(model, a.out)
    print(f"[extract] 写出 {path}")
    print("[extract] 复核要点：")
    print("  1) root 应该是 base_link；")
    print("  2) 每个 *_wheel_joint 的子刚体应是 *_wheel_link；")
    print("  3) 可动关节数应等于仿真里的关节数（云台版 22 / 无云台版 20 左右）。")


if __name__ == "__main__":
    main()
