#!/usr/bin/env bash
set -o pipefail
cd /home/simonelusetti/lion
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
mkdir -p outputs results/independent_judge
.venv/bin/python -B -m lion.independent_judge.run 2>&1 | tee outputs/independent_judge.log
printf '%s\n' "${PIPESTATUS[0]}" > results/independent_judge/exit_status.txt
