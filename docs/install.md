# 安装与运行

## 环境

Linux x86_64，RTX 显卡（训练）。Isaac Sim 5.1 + Isaac Lab 2.3.2，与 `wheeled-legged_RL` 同一套环境。
`python` 能用 `import isaaclab` 即可，不需要 Docker，也不需要 Newton / MuJoCo。

资产路径：

```bash
export WHEELBIPE_ASSET_ROOT=/path/to/wheeled-legged_RL/source/agent_world/agent_world/assets/usd_files
```

## 标定

```bash
python tools/calibrate.py --headless
```

依次看 6 段输出：

1. `joint_names` 顺序，无云台的版本应该是 18 - 20 个关节，`na=6`。
2. `FK 校验` 误差应 <= 1e-2 m。超过 2 cm 说明运动学树抽错了，`no_fork` 会算错，可以先加 `--no_kin`。
3. `FK 估计站立高度`，后面所有高度指令都相对它。
4. `零动作静置`：机身持续下沉、轮子不动，说明弹簧力方向或大小不对，先别训练。
5. 用稳态高度定 `--height_range` 和 `--term_height`。

## 训练

```bash
# 带行为克隆热启动（推荐）
python supermimic_wheelbipe/train.py --num_envs 4096 --headless \
  --log_dir logs/wb_main --lin_vel_x -2.7 2.7 --ang_vel_z -6.2832 6.2832

# 从零开始，前 1000 轮的想象数据基本是随机乱蹬
python supermimic_wheelbipe/train.py --num_envs 4096 --headless --log_dir logs/wb_scratch

# 或
./run_train.sh wb_main
```

最终配置（本仓库那次是续训约 4300 轮 / 43 分钟）：

```bash
python supermimic_wheelbipe/train.py --headless \
  --num_envs 4096 --iterations 24000 --wm_only_iters 800 --warmup_steps 1500 \
  --log_every 10 --horizon 16 --compile 0 --utd 2.0 \
  --wm_batches 5 --wm_batch 8192 --pi_batch 4096 \
  --lin_vel_x -1.5 1.5 --ang_vel_z -10.21 10.21 --height_range 0.20 0.28 \
  --log_dir logs/wb_final
```

启动时有两件事会打印出来：

- 零动作静置 `--settle_steps` 步（默认 120 步 = 2.4 s）取稳态根高度，作为重置高度和高度指令的基准。
  `[settle]` 行倾角 cos < 0.9 表示零动作下机器人会倒，先检查弹簧和执行器。已知数值可用 `--z_stand` 直接给。
- 高度指令窗口默认 `[z-0.08, z-0.02]`，用 `--height_range` 覆盖。策略稳了以后再往下扩，一次 0.02 m。

`evaluate.py` 和 `tools/play.py` 也会做同样的 settle，保证评测高度落在训练分布内。

前 `--wm_only_iters` 轮只训世界模型。`logs/<run>/train_log.csv` 每 50 轮一行：

| 列 | 含义 | 期望 |
|---|---|---|
| `wm_loss` / `wm_err8` | 世界模型误差 | 前 1000 轮快速下降后稳定 |
| `imag_survival` | 想象 rollout 存活率 | 长期 < 0.3 说明策略没学会或世界模型太差 |
| `sim_rew` | 仿真归一化步奖励 | 1.0 是理论上限 |
| `ep_len` / `ep_ret_norm` / `term_frac` | 仿真里的真实表现 | `term_frac` 高说明重置就摔或奖励不一致 |

## 评测

```bash
python supermimic_wheelbipe/evaluate.py --ckpt logs/wb_main/ckpt_004000.pt --headless
python supermimic_wheelbipe/evaluate.py --run_dir logs/wb_main --watch --headless
```

16 条指令：7 档直行，6 档原地转，直行带转向，后退带转向，站立。
报告 `norm_return`、失败数、`mean|e_lin| / mean|e_ang| / mean|e_h|`。

## 录制

一次 Isaac 启动，按指令逐段录，numpy 拼 2x2 存 mp4，不需要 ffmpeg 拼接。

```bash
# 默认 4 条速度 0.5 / 1.0 / 1.5 / -0.5
python tools/play.py --ckpt logs/wb_main/ckpt_004000.pt --enable_cameras

# 老师当参考
python tools/play.py --teacher auto --enable_cameras --video videos/teacher.mp4

# 四格放前进 + 转向，--commands 与 --wzs 一一对应
python tools/play.py --teacher auto --commands 0.8,0.8,-0.8,-0.8 --wzs 0,1.5,0,-1.5

# 变腿，四格相位自动错开 0.5 s
python tools/play.py --ckpt logs/wb_final/last.pt --commands 0,0,0,0 \
  --leg_lo 0.20 --leg_hi 0.28 --leg_period 2.0 --enable_cameras --video videos/leg.mp4

# 只看窗口不录
python tools/play.py --teacher auto --video none --cmd 1.0 0 -1
```

录制走 Replicator 的 render product（自建 `/World/rec_cam`），不依赖视口，ssh 也能录。
`--resolution 640 360` 拼出来是 1280x720，能被 16 整除。摔了的那格会冻结最后一帧补齐时间轴。
`--stills` 存每格最后一张 png。结束时打印 `scp` 命令，照抄即可。mp4 存不下来会退化成 png 序列。

## 工具

| 脚本 | 作用 | 需要 RTX |
|---|---|---|
| `tools/selftest_cpu.py` | 形状和逻辑自检 | 否 |
| `tools/extract_model.py` | 单独提取运动学树 | 否 |
| `tools/calibrate.py` | FK 校验、站立高度、弹簧力 | 是 |
| `tools/run_teacher.py` | 老师自检、录参考轨迹、行为克隆 | 是 |
| `tools/play.py` | 回放和录四格视频 | 是 |
| `supermimic_wheelbipe/evaluate.py` | 16 条指令评测 | 是 |
| `supermimic_wheelbipe/train.py` | 训练主循环 | 是 |
