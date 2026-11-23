#!/usr/bin/env bash
# usage: tools/bright/run_bright_eval.sh <task|env> <log> [extra x2real_eval args...]
# Runs one BRIGHT task env (board bright_arm6a_tabletop) against the policy
# service on 127.0.0.1:${PS_PORT:-8001}. Example zero-action smoke:
#   tools/bright/run_bright_eval.sh stack_rings /tmp/rings.log --timeout-s 20
set -u
cd "$(dirname "$0")/../.."
T="$1"; LOG="$2"; shift 2
case "$T" in *_id_env) ENV_NAME="$T";; *) ENV_NAME="artixon_arm6a_G_bright_${T}_id_env";; esac
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"   # use this checkout's manaenv (the conda env has an editable install of another one)
export MANAENV_ASSET_ROOT="${MANAENV_ASSET_ROOT:-$PWD/x2real-assets}"
P="${BRIGHT_PROXY:-http://mininas:8765}"
export http_proxy=$P https_proxy=$P HTTP_PROXY=$P HTTPS_PROXY=$P
export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate manaenv
case "$ENV_NAME" in *_cloth_*) BOARD=${BOARD:-bright_arm6a_cloth};; *) BOARD=${BOARD:-bright_arm6a_tabletop};; esac
python manaenv/scripts/x2real_eval.py --board "$BOARD" \
  --env "$ENV_NAME" --num-episodes "${EPISODES:-1}" --num-parallel-envs 1 \
  --policy-config policy/x2real_ps@v1.0.0.yaml \
  --server-address 127.0.0.1 --server-port "${PS_PORT:-8001}" --enable_cameras \
  --save-path "${SAVE_PATH:-./outputs/bright_eval/${T}}" "$@" > "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
