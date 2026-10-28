# SPDX-License-Identifier: MIT
"""速度指令库：替代原版的 ``MotionLib``（参考动作库）。"""

from __future__ import annotations

import torch

BIN_FRAMES = 50          # 1 s = 50 个控制步
GOAL_PER_FRAME = 4       # (vx, vy, wz, height)
CMD_DIM = GOAL_PER_FRAME


def eval_pairs():
    """评测/自检用的 (vx, wz) 指令网格：7 档直行 + 6 档原地转 + 3 个组合。"""
    vx = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    wz = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
    pairs = [(v, 0.0) for v in vx]
    pairs += [(0.0, w) for w in wz]
    pairs += [(1.0, 1.0), (-1.0, -1.0), (0.0, 0.0)]
    return pairs


class CommandLib:
    """指令"动作库"。字段语义与 ``supermimic.MotionLib`` 对应。"""

    def __init__(self, cmd: torch.Tensor, clip_lens, device: str = "cpu", names: list[str] | None = None):
        assert cmd.dim() == 2 and cmd.shape[1] == CMD_DIM
        self.device = device
        self.cmd = cmd.to(device)
        lens = [int(x) for x in clip_lens]
        self.clip_len = torch.tensor(lens, device=device)
        starts = torch.cumsum(torch.tensor([0] + lens[:-1], dtype=torch.long), 0)
        self.clip_start = starts.to(device)
        self.clip_end = self.clip_start + self.clip_len - 1
        self.num_frames = int(sum(lens))
        self.clip_id = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(lens)]).to(device)
        self.names = names or [f"clip{i}" for i in range(len(lens))]

        # 1 s bins（与超参表一致）
        bin_clip, bin_start = [], []
        for c, n in enumerate(lens):
            for b in range(int((n + BIN_FRAMES - 1) // BIN_FRAMES)):
                bin_clip.append(c)
                bin_start.append(int(self.clip_start[c]) + b * BIN_FRAMES)
        self.bin_clip = torch.tensor(bin_clip, device=device)
        self.bin_start = torch.tensor(bin_start, device=device)
        self.num_bins = len(bin_start)
        self.frame_bin = torch.zeros(self.num_frames, dtype=torch.long, device=device)
        for b in range(self.num_bins):
            c, s = bin_clip[b], bin_start[b]
            e = min(s + BIN_FRAMES, int(self.clip_end[c]) + 1)
            self.frame_bin[s:e] = b

    # 构造
    @staticmethod
    def random(
        device: str,
        num_clips: int = 256,
        clip_frames: int = 2000,
        seg_frames=(250, 750),
        lin_vel_x=(-2.0, 2.0),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(-6.0, 6.0),
        height=(0.20, 0.34),
        standing_prob: float = 0.1,
        seed: int = 0,
        focus: tuple | None = None,
        focus_prob: float = 0.4,
    ) -> "CommandLib":
        """随机指令库。"""
        g = torch.Generator(device="cpu").manual_seed(seed)
        U = lambda lo, hi, n=1: (torch.rand(n, generator=g) * (hi - lo) + lo)  # noqa: E731
        rows, lens, names = [], [], []
        for c in range(num_clips):
            segs = []
            t = 0
            while t < clip_frames:
                n = int(torch.randint(seg_frames[0], seg_frames[1] + 1, (1,), generator=g))
                n = min(n, clip_frames - t)
                if torch.rand(1, generator=g).item() < standing_prob:
                    v = torch.zeros(CMD_DIM)
                    v[3] = U(*height).item()
                else:
                    lo_vx, hi_vx = lin_vel_x
                    if focus is not None and torch.rand(1, generator=g).item() < focus_prob:
                        lo_vx, hi_vx = float(focus[0]), float(focus[1])
                    v = torch.tensor(
                        [U(lo_vx, hi_vx).item(), U(*lin_vel_y).item(), U(*ang_vel_z).item(), U(*height).item()]
                    )
                segs.append((t, n, v))
                t += n
            block = torch.zeros(clip_frames, CMD_DIM)
            for (s, n, v) in segs:
                block[s : s + n] = v
            rows.append(block)
            lens.append(clip_frames)
            names.append(f"cmd{c:03d}")
        return CommandLib(torch.cat(rows), lens, device, names)

    @staticmethod
    def fixed(commands, device: str, frames: int = 501, names: list[str] | None = None) -> "CommandLib":
        """评测用：每条指令一条长度为 ``frames`` 的 clip（指令恒定，跑满即超时）。"""
        c = torch.as_tensor(commands, dtype=torch.float32).reshape(-1, CMD_DIM)
        n = c.shape[0]
        cmd = c.repeat_interleave(frames, dim=0)
        names = names or [f"vx{v[0]:+.1f}_wz{v[2]:+.1f}_h{v[3]:.2f}" for v in c.tolist()]
        return CommandLib(cmd, [frames] * n, device, names)

    # 查询
    def cmd_at(self, f: torch.Tensor) -> torch.Tensor:
        return self.cmd[f]

    def future_frames(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """f+1 .. f+K，夹在 clip 末帧内（指令不变，不会跳变）。"""
        end = self.clip_end[self.clip_id[f]]
        fut = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        return torch.minimum(fut, end.unsqueeze(-1))

    def goal(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """goal 特征 (B, K*4)。"""
        fut = self.future_frames(f, K)
        return self.cmd[fut].reshape(f.shape[0], -1)
