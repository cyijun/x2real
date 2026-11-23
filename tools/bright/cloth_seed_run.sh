#!/usr/bin/env bash
# usage: seedprep.sh <seed> <template program>  -> runs/seed<seed>/program.py from a 3 s layout-only episode
cd ~/workspace/x2real-cloth/.cloth_tmp
S=$1; T=$2; L=runs/layout$S; R=runs/seed$S
mkdir -p $L $R
cp runs/smoke/program.py $L/program.py
printf "{\"program\": \"%s/%s/program.py\", \"run_dir\": \"%s/%s\", \"episode_len_s\": 3, \"vla\": \"null\"}\n" $PWD $L $PWD $L > state/active.json
rm -f $L/cloth.jsonl $L/run.log
MANA_CLOTH_TRACE=$PWD/$L/cloth.jsonl PS_PORT=8022 ./run_cloth.sh fold_towel_cloth $PWD/$L --timeout-s 3 --seed $S
python3 mkprog.py $L/cloth.jsonl $T $R/program.py basket
printf "{\"program\": \"%s/%s/program.py\", \"run_dir\": \"%s/%s\", \"episode_len_s\": 80, \"vla\": \"null\"}\n" $PWD $R $PWD $R > state/active.json
rm -f $R/cloth.jsonl $R/run.log
MANA_CLOTH_TRACE=$PWD/$R/cloth.jsonl PS_PORT=8022 ./run_cloth.sh fold_towel_cloth $PWD/$R --timeout-s 80 --seed $S
echo SEED_DONE $S
