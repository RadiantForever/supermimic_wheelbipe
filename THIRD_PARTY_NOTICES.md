# 第三方声明

本仓库代码以 [MIT](LICENSE) 发布。以下上游项目同为 MIT，按 MIT 要求保留其版权声明。

## 1. wheeled-legged_RL

- 仓库：https://github.com/scutrobotlab/wheeled-legged_RL
- 版权：Copyright (c) 2026 SCUTRobotLab
- 许可：MIT，全文见 https://github.com/scutrobotlab/wheeled-legged_RL/blob/main/LICENSE
- 使用范围：关节分组、执行器参数（DM8009 转子惯量、轮子 `effort_limit`）、奖励权重与 sigma，
  均取自该仓库的 `wheelbipe_V14` 配置。
- 本仓库不重分发其 USD 资产与预训练权重（`pretrained/26_infantry/*.pt`），需自行 clone 上游。

## 2. supermimic_vibed

- 仓库：https://github.com/zita-ch/supermimic_vibed
- 版权：Copyright (c) 2026 ETH Zurich, Chong Zhang
- 许可：MIT，全文见 https://github.com/zita-ch/supermimic_vibed/blob/main/LICENSE
- 使用范围：可微世界模型 + 想象 rollout + BPTT 的训练范式与部分代码结构。
- 本仓库不重分发其 LAFAN1 动作数据与 G1 资产。

## 3. SuperMimic 论文

Q. Liao, Y. Shao, J. Long, K. Zakka, R. Horowitz, P. Abbeel, K. Sreenath,
*SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*,
CoRL 2026, https://arxiv.org/abs/2610.28480。

本仓库是独立实现，与论文作者无隶属关系。引用请引论文。

## 4. 外部依赖

Isaac Sim 与 Isaac Lab（BSD-3-Clause）为外部依赖，不随本仓库分发。
