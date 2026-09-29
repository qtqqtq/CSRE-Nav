#!/usr/bin/env bash
# Compatibility wrapper for a single full benchmark run.
set -euo pipefail
if [[ $# != 2 ]]; then
    echo "usage: $0 cspace|neupan output.json" >&2
    exit 2
fi
PLANNER=$1
OUT=$(realpath -m "$2")
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source /opt/ros/noetic/setup.bash
source "$ROOT/catkin_ws/devel/setup.bash"
mkdir -p "$(dirname "$OUT")"
export PLANNER_BENCH_REPORT_DIR=$(dirname "$OUT")
python3 "$ROOT/evaluate/run_bench.py" "$PLANNER"
DEFAULT="$PLANNER_BENCH_REPORT_DIR/${PLANNER}_run.json"
if [[ $DEFAULT != "$OUT" ]]; then
    mv "$DEFAULT" "$OUT"
fi
