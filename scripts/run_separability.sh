#!/usr/bin/env bash
set -o pipefail
cd /home/simonelusetti/lion
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p results/separability outputs
.venv/bin/python -B -m lion.separability.run "$@" 2>&1 | tee -a outputs/separability.log
printf '%s\n' "${PIPESTATUS[0]}" > results/separability/exit_status.txt
