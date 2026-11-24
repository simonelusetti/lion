#!/usr/bin/env bash
set -o pipefail
cd /home/simonelusetti/lion
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p outputs results/fewnerd_full_grid
.venv/bin/python -B -m lion.label_change_grid.full_fewnerd 2>&1 | tee outputs/fewnerd_full_grid.log
printf '%s\n' "${PIPESTATUS[0]}" > results/fewnerd_full_grid/exit_status.txt
