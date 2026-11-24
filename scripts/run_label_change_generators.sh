#!/usr/bin/env bash
set -o pipefail
cd /home/simonelusetti/lion
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p outputs results/label_change_generators
.venv/bin/python -B -m lion.label_change.generators 2>&1 | tee outputs/label_change_generators.log
printf '%s\n' "${PIPESTATUS[0]}" > results/label_change_generators/exit_status.txt
