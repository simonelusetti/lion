#!/usr/bin/env bash
set -o pipefail
cd /home/simonelusetti/lion
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p outputs results/population_quality
.venv/bin/python -B -m lion.guided_refinement.population 2>&1 | tee outputs/population_quality.log
printf '%s\n' "${PIPESTATUS[0]}" > results/population_quality/exit_status.txt
