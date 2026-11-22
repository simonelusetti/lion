#!/usr/bin/env bash
set -o pipefail
cd /home/simonelusetti/lion
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
.venv/bin/python -B -m lion.deprel_replication.run 2>&1 | tee results/deprel_replication/run.log
printf '%s\n' "${PIPESTATUS[0]}" > results/deprel_replication/exit_status.txt
exec bash
