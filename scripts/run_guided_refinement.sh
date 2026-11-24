#!/usr/bin/env bash
set -o pipefail
cd /home/simonelusetti/lion
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p outputs results/guided_refinement
.venv/bin/python -B -m lion.guided_refinement.run 2>&1 | tee outputs/guided_refinement.log
printf '%s\n' "${PIPESTATUS[0]}" > results/guided_refinement/exit_status.txt
