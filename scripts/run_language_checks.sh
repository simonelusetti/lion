#!/usr/bin/env bash
set -o pipefail
cd /home/simonelusetti/lion
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
mkdir -p outputs results/language_checks
.venv/bin/python -B -m lion.language_checks.run 2>&1 | tee outputs/language_checks.log
printf '%s\n' "${PIPESTATUS[0]}" > results/language_checks/exit_status.txt
