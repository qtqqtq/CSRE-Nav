#!/bin/bash
# Run the benchmark end to end.
#
#   run_bench.sh cspace      # one planner
#   run_bench.sh neupan
#   run_bench.sh both        # cspace, reset, neupan, then report.py
#
# Extra arguments are passed through to roslaunch, so the negative control is
#
#   run_bench.sh both goal_set:=$PWD/config/goal_sets/negcontrol.json
#
# Everything the harness writes goes to $REPORT_DIR (default reports/).  An
# existing report is parked as <name>.json.prev rather than deleted, so a run
# that dies can be told apart from one that never happened.
#
# On the timing: one trial is up to --trial-timeout sim seconds, and sim time
# runs at roughly wall-clock rate headless, so budget ~90 s per goal plus ~20 s
# of startup.  The waits below are generous on purpose -- a bench run that is
# merely slow is not a failure, and killing it at a tight deadline produces a
# truncated report that is worse than no report.
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Resolve the package through rospack when it is available, because the two
# layouts differ: from the source tree this script sits in <pkg>/scripts/, but
# installed it sits in <ws>/devel/lib/planner_bench/ and `$HERE/..` is the lib
# directory, which would put reports/ and the workspace root in the wrong place.
PKG=$(rospack find planner_bench 2>/dev/null || true)
[ -n "$PKG" ] || PKG=$(cd "$HERE/.." && pwd)
# Overridable so a control run cannot overwrite the benchmark's own numbers.
# Note bench.launch has its own `report_dir` arg defaulting to <pkg>/reports --
# the value below is passed through to it explicitly, so setting the environment
# variable actually moves the output rather than just this script's idea of it.
REPORT_DIR="${PLANNER_BENCH_REPORT_DIR:-$PKG/reports}"
LOG_DIR="${PLANNER_BENCH_LOG_DIR:-${TMPDIR:-/tmp}/planner_bench}"
WS=$(cd "$PKG/../.." && pwd)          # catkin_ws

mkdir -p "$LOG_DIR" "$REPORT_DIR"

run_one() {
    local planner="$1"; shift
    local out="$REPORT_DIR/${planner}_run.json"
    local log="$LOG_DIR/${planner}_launch.log"

    echo "--- $planner ---"
    bash "$HERE/reset_ros.sh" || { echo "RESET FAILED -- not starting a run on a dirty master"; return 3; }

    # Park, do not delete.  If this run dies, the previous numbers are still
    # there to compare against, and "no new data" is distinguishable from
    # "the old data was thrown away".
    if [ -s "$out" ]; then
        mv -f "$out" "$out.prev" && echo "previous report -> $(basename "$out").prev"
    fi

    # setsid: detach from this shell's process group so a Ctrl-C on the driver
    # does not take the simulation with it mid-trial.
    setsid roslaunch planner_bench bench.launch planner:="$planner" gui:=false \
        report_dir:="$REPORT_DIR" "$@" \
        > "$log" 2>&1 &
    echo "launched, log: $log"

    # Wait for the harness to APPEAR before watching for it to vanish.  roslaunch
    # returns as soon as it has spawned the nodes, so a disappearance check on
    # the first iteration fires before bench.py has started -- reporting a
    # failure that never happened.
    local n
    for n in $(seq 1 300); do
        pgrep -f "bench.py --planner $planner" >/dev/null && break
        sleep 1
    done
    if ! pgrep -f "bench.py --planner $planner" >/dev/null; then
        echo "the harness never came up -- check $log"
        tail -c 2000 "$log"
        return 1
    fi
    echo "harness up after ${n}s"

    for n in $(seq 1 5400); do
        [ -s "$out" ] && { echo "report after ${n}s"; break; }
        if ! pgrep -f "bench.py --planner $planner" >/dev/null; then
            echo "the harness exited without writing a report -- check $log"
            tail -c 2000 "$log"
            return 1
        fi
        sleep 1
    done
    [ -s "$out" ] || { echo "gave up waiting for $out"; return 1; }

    sleep 3
    python3 - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for t in d["trials"]:
    print("  goal %-3d %-9s final_error=%-8s %s"
          % (t["i"], t["verdict"],
             "-" if t.get("final_error") is None else round(t["final_error"], 3),
             t["reason"][:64]))
PY
    return 0
}

if [ $# -lt 1 ]; then
    echo "usage: $(basename "$0") cspace|neupan|both [extra roslaunch args]" >&2
    exit 2
fi
WHICH="$1"; shift

case "$WHICH" in
    cspace|neupan) run_one "$WHICH" "$@" ;;
    both)
        run_one cspace "$@" || exit $?
        run_one neupan "$@" || exit $?
        echo
        echo "--- comparison ---"
        python3 "$HERE/report.py"
        ;;
    *) echo "unknown planner: $WHICH (want cspace, neupan or both)" >&2; exit 2 ;;
esac
