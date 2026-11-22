#!/usr/bin/env bash
set -o pipefail
cd /home/simonelusetti/lion
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p results/label_share outputs
.venv/bin/python -B -m lion.label_share.run "$@" 2>&1 | tee -a outputs/label_share.log
printf '%s\n' "${PIPESTATUS[0]}" > results/label_share/exit_status.txt
