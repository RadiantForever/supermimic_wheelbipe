#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# 用法: ./run_train.sh <run_name> [额外的 train.py 参数...]
#   CPUS=0-23 ./run_train.sh wb_main --num_envs 4096
# 日志: logs/<run_name>/train_log.csv  ckpt_*.pt
set -euo pipefail
cd "$(dirname "$0")"

NAME=${1:-wb_main}
shift || true
SESSION="wb_${NAME}"

CMD="python supermimic_wheelbipe/train.py --headless --log_dir logs/${NAME} $*"
if [ -n "${CPUS:-}" ]; then
  CMD="taskset -c ${CPUS} ${CMD}"
fi

mkdir -p "logs/${NAME}"   # tee 会立刻写这个目录，必须先建好，否则首次运行会断管道
if command -v tmux >/dev/null 2>&1; then
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
  tmux new -d -s "${SESSION}" "cd $(pwd) && ${CMD} 2>&1 | tee logs/${NAME}/stdout.log"
  echo "started tmux session '${SESSION}'"
  echo "  attach : tmux attach -t ${SESSION}"
  echo "  eval   : python supermimic_wheelbipe/evaluate.py --run_dir logs/${NAME} --watch --headless"
else
  eval "${CMD}"
fi
