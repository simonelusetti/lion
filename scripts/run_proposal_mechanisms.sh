#!/usr/bin/env bash
set -o pipefail
cd /home/simonelusetti/lion
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p outputs results/proposal_mechanisms
.venv/bin/python -B -m lion.proposal_mechanisms.run 2>&1 | tee outputs/proposal_mechanisms.log
printf '%s\n' "${PIPESTATUS[0]}" > results/proposal_mechanisms/exit_status.txt
