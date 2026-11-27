#!/usr/bin/env bash
set -euo pipefail
cd /home/simonelusetti/lion
export USE_TF=0 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
prior_runner_pid="$1"
run_dir=results/independent_judge
echo 'Waiting for the existing judge-training stage to finish before changing runners.'
while [[ ! -f "$run_dir/judge_train/runtime.json" ]]; do
    kill -0 "$prior_runner_pid"
    sleep 2
done
if kill -0 "$prior_runner_pid" 2>/dev/null; then
    kill -TERM "$prior_runner_pid"
    while kill -0 "$prior_runner_pid" 2>/dev/null; do sleep 1; done
fi
rm -f "$run_dir/exit_status.txt"
trap 'printf "%s\n" "$?" > "$run_dir/exit_status.txt"' EXIT
cp "$run_dir/benchmark/estimate.json" "$run_dir/benchmark/estimate_before_final_only.json"
for stage in benchmark generate judge_outputs certificate train test report; do
    .venv/bin/python -B -m lion.independent_judge.run "stage.experiment=$stage" 2>&1 | tee -a outputs/independent_judge.log
done
