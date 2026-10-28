# 与上游的差异、老师策略、奖励

## 1. 老师策略用在哪

同一台机器人在 `wheeled-legged_RL` 里已有训好的 PPO 策略，本仓库拿它当 teacher，只有三处用到，都能关掉。

| 用途 | 开关 | 本仓库实际用量 |
|---|---|---|
| 世界模型热身：buffer 里混一部分老师轨迹 | `--teacher_mix` + `--teacher_mix_iters` | 0.3，前 800 轮 |
| 行为克隆热启动 | `--init_policy logs/bc/bc_init.pt` | 没开 |
| 环境自检 | `--mode play` / `--mode play --grid` | 只是诊断 |

没有做的事：

- 不从老师权重初始化网络。`wm/pi/critic` 全随机，结构也不同（老师 35 维 obs，256-128-64；本仓库 78 维 obs）。
- 不做 logit 蒸馏，老师不进损失函数。
- 没做过「完全不给老师数据」的对照，所以不能声称不用老师也能到这个精度。

老师权重不随仓库分发。要自己 clone `wheeled-legged_RL` 并指向 `pretrained/26_infantry/`。

### 1.1 老师接口

`teacher.py` 复刻 `wheelbipe25_v3/env.py::_get_observations` 的 35 维单帧 obs：

| idx | 内容 | 处理 |
|---|---|---|
| 0-2 | `lin_vel_x, lin_vel_y, ang_vel_z` | 指令，`ang_vel_z x 0.5` |
| 3 | `height_cmd` | `clip[0,1] x 5.0` |
| 4-6 | `root_ang_vel_b` | `x 0.5` |
| 7-9 | `projected_gravity_b` | 原样 |
| 10-15 | 6 个主动关节的 `q - default`，轮子恒 0 | 原样 |
| 16-19 | 腿关节速度 | `x 0.1` |
| 20-21 | 轮速 | `x 0.1` |
| 22-27 | 上一步动作 | 原样 |
| 28-34 | `ctrl_mode_obs` = `[1,0,0,0,0,0,0]` | `[1,1,1,1,1,5,1]` |

动作 6 维，语义与本仓库一致：`q_leg = default + 0.5*a[:4]`，`w_wheel = 10*a[4:]` rad/s，`use_wheel_vel_control: true`。

验证方式是从 `model_8000.pt` 的 `actor.*` 重建 MLP，与 `exported/policy.pt` 在同一批随机输入上比较，输出差 0.0。
`tools/selftest_cpu.py` 里另有一条断言，要求姿态变化必须引起动作变化。

### 1.2 用法

```bash
python tools/run_teacher.py --list                       # 看有哪些老师，纯 CPU

python tools/run_teacher.py --mode play --grid --teacher auto --headless --seconds 10
python tools/run_teacher.py --mode play --teacher auto --cmd 1.0 0 -1 --enable_cameras

python tools/run_teacher.py --mode bc --num_envs 2048 --out logs/bc/bc_init.pt

python supermimic_wheelbipe/train.py --headless --log_dir logs/wb_main \
  --init_policy logs/bc/bc_init.pt \
  --teacher_ckpt "$(python -c 'from supermimic_wheelbipe.teacher import find_pretrained as f; print([r for r in f() if "flat_and_rotation" in r][-1])')" \
  --teacher_mix 0.5 --teacher_mix_iters 3000
```

`--mode play --grid` 先跑，老师走不起来说明环境移植错了，别去调算法。
`--mode record` 把老师 rollout 录成 npz（`p/R/v/w/q/qd/cmd/a`，50 Hz），就是原版意义上的参考动作库。

`ACTION_SCALE_WHEEL=10.0`、`lin_vel_x=(-2.7,2.7)`、`ang_vel_z=(-6.28,6.28)` 都是对着 `pretrained/.../params/env.yaml` 校准的。

## 2. 与论文 / 上游的偏差

| 论文或上游 | 本仓库 | 原因 |
|---|---|---|
| 不用预训练策略 | 老师参与世界模型前 800 轮混采，可选 BC 热启动 | 轮腿接触硬，世界模型起手就发散 |
| 终止惩罚 -200 | -20，并对 bin 采样的 `G/G_max` 加夹紧 | -200 让 softmax 权重爆 40 倍 |
| 想象 rollout 不加约束 | 每步 `mdp.sanitize()` 夹限位、速度、高度 | 轮腿 20 步后易发散，正常区域梯度不受影响 |
| LAFAN1 参考动作库 | 换速度指令库 `CommandLib` | 轮腿没有 mocap，clip / bin / 重置采样 / goal 前视的结构保留 |
| URDF 转 MJCF 再可微 FK | 从 USD 的 `UsdPhysics` 直接提运动学树，配纯 torch FK | 只有 PhysX USD，没有 URDF 和 STL |
| 29 关节全受控 | 动作 6 维：4 腿位置 + 2 轮速度，云台排除 | 硬件只有 6 个主动关节 |
| 奖励由仿真器算 | 解析式奖励 | 轮腿的奖励项全是刚体量，可闭式求值 |
| 29 维状态含轮角 | 去掉轮角只留轮速，`state_dim=47` | 轮角无界，转速可到 60 rad/s |

## 3. 奖励对应

权重和 sigma 取自 `wheelbipe_V14/env_cfg.py` 的 `rewards` 与 `RewardCfgV14`，实现在 `mdp.py::reward`。

| 上游 key | 权重 | 本仓库 |
|---|---|---|
| `track_lin_vel_xy` | +1.0 | `exp(-e^2/0.5)`，e 是水平前向速度（按 pitch 投影） |
| `track_lin_vel_xy_square` | -1.0 | `(0.25*e)^2` |
| `track_ang_vel_z` / `_square` | +1.0 / -1.0 | `exp(-e^2/0.25)`、`(0.5*e)^2` |
| `track_height_exp_tight` / `_square` | +1.0 / -1.0 | `exp(-e^2/1e-3)`、`(10*e)^2`，高度取根高度 |
| `flat_orientation_{x,y}_exp` | +1.0 | `exp(-g_b^2/sigma)`，sigma = 0.01 / 0.02 |
| `flat_orientation_{x,y}_v` | -2.0 | `((2e^{-vx^2/3}+2)*g_b)^2` |
| `lin_vel_z` / `ang_vel_xy` | -0.5 / -0.05 | 同 |
| `action_rate` | -0.01 | `\|a-a_prev\|^2` |
| `leg_joint_vel` / `wheel_vel` | -5e-3 / -1e-5 | 同 |
| `joint_torque` / `wheel_power` | -1e-4 / -1e-4 | 隐式 PD 解析算 `tau = kp(q*-q) - kd*qd` |
| `no_fork` / `no_fork_square` | -1.0 | 左右轮心前后错开量 `dx`，`(5*max(\|dx\|-0.05,0))^2` |
| `termination` | -200 → -20 | 见第 2 节 |

终止条件：根高度低于 `term_height`（约等于 base_link 触地）、倾角 cos < cos40°、速度过大、NaN。

`no_fork` 需要 FK。第一次训练会写 `data/wheelbipe_v14_2_model.json`，先跑 `calibrate.py` 确认误差，不行就 `--no_kin` 关掉。

## 4. 排错

| 现象 | 先查什么 |
|---|---|
| `wm_err8` 一直大于 1 | `calibrate` 的静置过程是否稳定，再看 `--wm_batches` / `--wm_window`，最后把 `--horizon` 从 24 降到 16 |
| `imag_survival` 长期很低 | 世界模型在终止边界附近不准，加大 `--wm_only_iters`，或放宽 `term_height` |
| 仿真 `term_frac` 高但想象里正常 | `mdp` 改了却没走 `sanitize`，或者 `z_stand` 设错导致重置就摔 |
| 轮子转但不前进 | `--friction`、轮子 `effort_limit=5 N·m`、`ACTION_SCALE_WHEEL` 是否过大 |
| 腿立不住 | 弹簧没生效，`calibrate` 里 `spring force @ q=0` 应该是 ~570 N；或者 `term_height` 太高 |
| FK 误差 1e-1 量级 | USD 里的 `physics:axis` / `localPos0/1` 与仿真不一致，跑 `tools/extract_model.py` 打印树结构核对 |
