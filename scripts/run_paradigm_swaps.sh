#!/usr/bin/env bash
set -o pipefail
cd /home/simonelusetti/lion
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
mkdir -p results/paradigm_swaps
.venv/bin/python -B -m lion.paradigm_swaps.run 2>&1 | tee results/paradigm_swaps/run.log
printf '%s\n' "${PIPESTATUS[0]}" > results/paradigm_swaps/exit_status.txt
exec bash
