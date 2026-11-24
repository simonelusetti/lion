#!/usr/bin/env bash
set -o pipefail
cd /home/simonelusetti/lion
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p outputs results/mechanism_training_pilot
.venv/bin/python -B -m lion.proposal_mechanisms.train_pilot 2>&1 | tee outputs/mechanism_training_pilot.log
printf '%s\n' "${PIPESTATUS[0]}" > results/mechanism_training_pilot/exit_status.txt
