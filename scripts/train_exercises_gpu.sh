#!/usr/bin/env bash
# Run the original fixed E1-E3 protocol; test opens only after all train runs.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
course_python="${COURSE_PYTHON:-python3}"
mkdir -p logs
"$course_python" -m scripts.prepare_data --data-root data
"$course_python" -m scripts.run_exercises_isolated --group all --seeds 42 43 44 --device cuda --num-workers 8 --resume
"$course_python" -m scripts.run_exercises_isolated --group all --seeds 42 43 44 --phase test --confirm-test --device cuda --num-workers 8
for course_seed in 42 43 44; do
  "$course_python" -m E3.diagnose --run "runs/e3_lstm_rows/seed_${course_seed}" --device cuda
  "$course_python" -m E3.diagnose --run "runs/e3_lstm_rows_combined/seed_${course_seed}" --device cuda
done
"$course_python" -m scripts.export_report
date -u +%FT%TZ >logs/exercises.complete
