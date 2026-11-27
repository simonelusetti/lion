#!/usr/bin/env bash
set -o pipefail
cd /home/simonelusetti/lion
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
mkdir -p outputs results/matched_regime
.venv/bin/python -B -m lion.matched_regime.run 2>&1 | tee outputs/matched_regime.log
printf '%s\n' "${PIPESTATUS[0]}" > results/matched_regime/exit_status.txt
