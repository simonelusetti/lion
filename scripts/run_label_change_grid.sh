#!/usr/bin/env bash
set -o pipefail
cd /home/simonelusetti/lion
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p outputs results/label_change_grid
.venv/bin/python -B -m lion.label_change_grid.run 2>&1 | tee outputs/label_change_grid.log
printf '%s\n' "${PIPESTATUS[0]}" > results/label_change_grid/exit_status.txt
