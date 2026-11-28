#!/usr/bin/env bash
cd /home/simonelusetti/lion
mkdir -p results/information_audit
USE_TF=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m lion.information_audit.run > results/information_audit/run.log 2>&1
status=$?
printf '%s\n' "$status" > results/information_audit/exit_status.txt
exec bash
