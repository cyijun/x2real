#!/usr/bin/env bash
# Serve wall-oss-0.5 with the vla_debug fixes on 127.0.0.1:${PORT:-8011}.
# env knobs: WALLX_GRIPPER_MODEL_SCALE (deploy default 5.5), WALLX_PREOPEN_CHUNKS (deploy default 1), WALLX_DUMP_PATH (jsonl of state+chunk per inference), WALLX_AGENT_NO_PATCH=1
set -u
source ~/miniforge3/etc/profile.d/conda.sh
conda activate wallx
export HF_ENDPOINT=https://hf-mirror.com
export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
V=$HOME/workspace/x2real/vla_debug
export WALLX_CHECKPOINT_PATH=${WALLX_CHECKPOINT_PATH:-$V/ckpt_wall_oss_05_3cam}
export WALLX_ROOT=$HOME/workspace/wall-x
export WALLX_BACKEND=standard
cd ~/workspace/x2real/3rd/policy_space
exec python -u $V/ps_patched.py serve $V/policies/wallx_oss05_fix/deploy.yaml --host 127.0.0.1 --port "${PORT:-8011}"
