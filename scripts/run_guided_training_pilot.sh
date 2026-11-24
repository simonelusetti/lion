#!/usr/bin/env bash
set -o pipefail
cd /home/simonelusetti/lion
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p outputs results/guided_training_pilot
.venv/bin/python -B -m lion.guided_refinement.train_pilot 2>&1 | tee outputs/guided_training_pilot.log
printf '%s\n' "${PIPESTATUS[0]}" > results/guided_training_pilot/exit_status.txt
