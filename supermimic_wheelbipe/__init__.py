# SPDX-License-Identifier: MIT
"""SuperMimic 风格训练框架的轮腿（wheelbipe V14_2）适配版。"""

__version__ = "0.1.0"


def exit_fast(code: int = 0):
    """先 flush 再硬退出。"""
    import os
    import sys

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(code)
