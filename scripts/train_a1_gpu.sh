#!/usr/bin/env bash
# Run A1 independently after E1-E3 releases the single GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
course_python="${COURSE_PYTHON:-python3}"
mkdir -p artifacts/logs
"$course_python" -m scripts.prefetch_cifar100 --data-root data
"$course_python" -c 'from torchvision.datasets import CIFAR100; CIFAR100(root="data", train=True, download=True)'
"$course_python" -m A1.run_suite --seeds 42 43 44 --device cuda --num-workers 8 --resume
"$course_python" -m A1.run_suite --seeds 42 43 44 --phase test --confirm-test --device cuda --num-workers 8
"$course_python" -m A1.export_report
date -u +%FT%TZ >artifacts/logs/a1.complete
