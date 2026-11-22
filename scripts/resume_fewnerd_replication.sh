#!/usr/bin/env bash
set -o pipefail
cd /home/simonelusetti/lion
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
rm -f results/fewnerd_replication/exit_status.txt
for stage in train metadata generate measure analyse report; do
    .venv/bin/python -B -m lion.fewnerd_replication.run "stage.experiment=$stage" 2>&1 | tee -a results/fewnerd_replication/resume.log
    run_status=${PIPESTATUS[0]}
    if [ "$run_status" -ne 0 ]; then
        printf '%s\n' "$run_status" > results/fewnerd_replication/exit_status.txt
        exec bash
    fi
done
printf '0\n' > results/fewnerd_replication/exit_status.txt
exec bash
