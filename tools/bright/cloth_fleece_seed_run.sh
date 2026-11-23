#!/usr/bin/env bash
# usage: fleece_seed.sh <seed> <template program> <tag>  (layout from runs/layout<seed>, run in runs/<tag><seed>)
cd ~/workspace/x2real-cloth/.cloth_tmp
S=$1; T=$2; TAG=$3; L=runs/layout$S; R=runs/$TAG$S
mkdir -p $L $R
if ! grep -q "\"objects\"" $L/cloth.jsonl 2>/dev/null; then
  cp runs/smoke/program.py $L/program.py
  printf "{\"program\": \"%s/%s/program.py\", \"run_dir\": \"%s/%s\", \"episode_len_s\": 3, \"vla\": \"null\"}\n" $PWD $L $PWD $L > state/active.json
  rm -f $L/cloth.jsonl $L/run.log
  MANA_CLOTH_TRACE=$PWD/$L/cloth.jsonl PS_PORT=8022 ./run_cloth.sh fold_towel_cloth $PWD/$L --timeout-s 3 --seed $S
fi
python3 mkprog.py $L/cloth.jsonl $T $R/program.py basket
printf "{\"program\": \"%s/%s/program.py\", \"run_dir\": \"%s/%s\", \"episode_len_s\": 80, \"vla\": \"null\"}\n" $PWD $R $PWD $R > state/active.json
rm -f $R/cloth.jsonl $R/run.log
MANA_CLOTH_PROFILE=${MANA_CLOTH_PROFILE:-bright_fleece_30} MANA_CLOTH_TIMESTEP=${MANA_CLOTH_TIMESTEP:-runtime/timestep/cloth_480hz.yaml} \
  MANA_CLOTH_TRACE=$PWD/$R/cloth.jsonl PS_PORT=8022 ./run_cloth.sh fold_towel_cloth $PWD/$R --timeout-s 80 --seed $S
echo SEED_DONE $TAG $S
