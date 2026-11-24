#!/usr/bin/env bash
set -o pipefail
cd /home/simonelusetti/lion
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p results/label_change_fewnerd_diagnostic outputs
for stage in prepare scorer generate certify summarize; do
    .venv/bin/python -B -m lion.label_change.fewnerd_diagnostic "stage.experiment=$stage" 2>&1 | tee -a outputs/fewnerd_diagnostic.log
    run_status=${PIPESTATUS[0]}
    if [ "$run_status" -ne 0 ]; then
        printf '%s\n' "$run_status" > results/label_change_fewnerd_diagnostic/exit_status.txt
        exit "$run_status"
    fi
done
printf '0\n' > results/label_change_fewnerd_diagnostic/exit_status.txt
