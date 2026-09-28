#!/usr/bin/env bash
# Run the shared benchmark from source. Requires a sourced ROS Noetic workspace.
set -euo pipefail

BENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$BENCH_DIR/.." && pwd)
REPORT_DIR=${PLANNER_BENCH_REPORT_DIR:-$BENCH_DIR/reports}
LOG_DIR=${PLANNER_BENCH_LOG_DIR:-${TMPDIR:-/tmp}/planner_bench}
WORLD=${PLANNER_BENCH_WORLD:-$BENCH_DIR/worlds/limo_complex.world}
CONFIG=${PLANNER_BENCH_CONFIG:-$BENCH_DIR/config/bench.yaml}
GUI=${PLANNER_BENCH_GUI:-true}
ROS_PORT=${PLANNER_BENCH_ROS_PORT:-11511}
GAZEBO_PORT=${PLANNER_BENCH_GAZEBO_PORT:-11545}
export ROS_MASTER_URI="http://127.0.0.1:$ROS_PORT"
export GAZEBO_MASTER_URI="http://127.0.0.1:$GAZEBO_PORT"
MODE=${1:-}
shift || true

if [[ ! $MODE =~ ^(cspace|neupan|both)$ ]]; then
    echo "usage: $0 cspace|neupan|both [bench.py arguments]" >&2
    exit 2
fi
if ! command -v roslaunch >/dev/null || ! command -v rosrun >/dev/null; then
    echo "Source /opt/ros/noetic/setup.bash and catkin_ws/devel/setup.bash first." >&2
    exit 2
fi
read -r START_X START_Y START_YAW <<< "$(python3 -c 'import yaml,sys; print(*yaml.safe_load(open(sys.argv[1]))["arena"]["start"])' "$BENCH_DIR/config/bench.yaml")"
mkdir -p "$REPORT_DIR" "$LOG_DIR"

sim_pid=
planner_pid=
cleanup() {
    if [[ -n $planner_pid ]]; then
        kill -- "-$planner_pid" 2>/dev/null || true
        wait "$planner_pid" 2>/dev/null || true
        planner_pid=
    fi
    if [[ -n $sim_pid ]]; then
        kill -- "-$sim_pid" 2>/dev/null || true
        wait "$sim_pid" 2>/dev/null || true
        sim_pid=
    fi
}
trap cleanup EXIT INT TERM

run_one() {
    local planner=$1
    shift
    local out="$REPORT_DIR/${planner}_run.json"
    cleanup
    if rosparam get /run_id >/dev/null 2>&1; then
        echo "ROS master at $ROS_MASTER_URI is already running; choose another PLANNER_BENCH_ROS_PORT" >&2
        return 1
    fi
    if [[ -s $out ]]; then
        mv -f "$out" "$out.prev"
    fi

    # Use multi_robot_scenario's maintained robot and obstacle bringup, with
    # the benchmark's copied world. This starts Gazebo and ROS master together.
    setsid roslaunch -p "$ROS_PORT" multi_robot_scenario multi_robot_scenario.launch \
        world_name:="$WORLD" obstacles:=true gui:="$GUI" \
        x:="$START_X" y:="$START_Y" yaw:="$START_YAW" >"$LOG_DIR/sim.log" 2>&1 &
    sim_pid=$!
    local ready=false
    for _ in $(seq 1 120); do
        if ! kill -0 "$sim_pid" 2>/dev/null; then
            echo "Simulation stopped; see $LOG_DIR/sim.log" >&2
            return 1
        fi
        if rosservice list 2>/dev/null | grep -Fxq /gazebo/set_model_state; then
            ready=true
            break
        fi
        sleep 1
    done
    if [[ $ready != true ]]; then
        echo "Gazebo did not become ready; see $LOG_DIR/sim.log" >&2
        return 1
    fi

    setsid "$BENCH_DIR/start_planner.sh" "$planner" >"$LOG_DIR/${planner}.log" 2>&1 &
    planner_pid=$!
    sleep 2
    if ! kill -0 "$planner_pid" 2>/dev/null; then
        echo "Planner stopped; see $LOG_DIR/${planner}.log" >&2
        return 1
    fi

    echo "Running $planner against $(basename "$WORLD")"
    python3 "$BENCH_DIR/bench.py" --planner "$planner" --world "$WORLD" \
        --config "$CONFIG" --out "$out" "$@"
    echo "Wrote $out"
    cleanup
}

case "$MODE" in
    cspace|neupan) run_one "$MODE" "$@" ;;
    both)
        run_one cspace "$@"
        run_one neupan "$@"
        python3 "$BENCH_DIR/report.py" "$REPORT_DIR/cspace_run.json" "$REPORT_DIR/neupan_run.json"
        ;;
esac
