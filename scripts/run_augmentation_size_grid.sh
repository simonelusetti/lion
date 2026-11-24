#!/usr/bin/env bash
set -o pipefail
cd /home/simonelusetti/lion
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p outputs results/augmentation_size_grid
.venv/bin/python -B -m lion.guided_refinement.size_grid 2>&1 | tee outputs/augmentation_size_grid.log
printf '%s\n' "${PIPESTATUS[0]}" > results/augmentation_size_grid/exit_status.txt
