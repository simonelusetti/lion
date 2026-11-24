#!/usr/bin/env bash
set -o pipefail
cd /home/simonelusetti/lion
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p results/conflict outputs
for stage in features context supply room analyse report; do
    .venv/bin/python -B -m lion.conflict.run "stage.experiment=$stage" 2>&1 | tee -a outputs/conflict.log
    run_status=${PIPESTATUS[0]}
    if [ "$run_status" -ne 0 ]; then
        printf '%s\n' "$run_status" > results/conflict/exit_status.txt
        exit "$run_status"
    fi
done
printf '0\n' > results/conflict/exit_status.txt
