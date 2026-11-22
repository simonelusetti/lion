#!/usr/bin/env bash
set -o pipefail
cd /home/simonelusetti/lion
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p results/layer_sensitivity outputs
.venv/bin/python -B -m lion.layer_sensitivity.run "$@" 2>&1 | tee -a outputs/layer_sensitivity.log
printf '%s\n' "${PIPESTATUS[0]}" > results/layer_sensitivity/exit_status.txt
