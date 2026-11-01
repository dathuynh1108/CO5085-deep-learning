#!/usr/bin/env bash
# One GPU, fixed protocol, complete both suites before final report processing.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/logs
course_exit() {
  local course_status=$?
  printf "%s\n" "$course_status" >artifacts/logs/coursework.exit
}
trap course_exit EXIT
bash scripts/train_exercises_gpu.sh
bash scripts/train_a1_gpu.sh
course_python="${COURSE_PYTHON:-python3}"
"$course_python" -m scripts.write_findings
"$course_python" -m scripts.build_report
"$course_python" -m scripts.build_report --report-root reports/assignment1 --jobname assignment1
"$course_python" scripts/build_site.py
date -u +%FT%TZ >artifacts/logs/coursework.complete
