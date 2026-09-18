#!/bin/bash
# Kill every ROS/Gazebo process and PROVE it, before starting a run.
#
# Run this between trials.  A stale rosmaster or gzserver from the previous
# planner survives into the next run and the symptom is not "port in use" -- it
# is a second gzserver attached to the old master, a robot that does not move,
# and a report full of no_motion that looks exactly like a planner bug.  It cost
# a long time to learn that; this script exists so it is not learned twice.
#
# Two traps it avoids, both of which have actually bitten:
#
#  1. Not `ss`.  In this container `ss -ltn` prints nothing even while rosmaster
#     holds 11311, so a port check reports "free" no matter what and a stale
#     gazebo survives into the next run.
#
#  2. Patterns must be the real executable paths, not loose words, and the
#     caller's own process ancestry must be excluded.  A pattern like "[g]azebo"
#     still matches an unrelated caller whose arguments mention it (e.g. a world
#     path under limo_gazebo_sim), and pkill -9 then kills the caller -- observed
#     as exit 137.  The bracket trick only hides pkill's own argv.
#
# Exits 0 and prints "clean" when nothing survived, non-zero with the survivors
# listed otherwise.  Run this before trusting any run that follows it.
set -u

SELF=$$
protect=" $SELF "
p=$SELF
while [ -n "$p" ] && [ "$p" != "1" ] && [ "$p" != "0" ]; do
    p=$(awk '{print $4}' /proc/$p/stat 2>/dev/null)
    [ -n "$p" ] && protect="$protect $p "
done

PATTERNS=(
    "/opt/ros/noetic/bin/roslaunch"
    "/opt/ros/noetic/bin/rosmaster"
    "/opt/ros/noetic/bin/rosout"
    "/opt/ros/noetic/lib/gazebo_ros/gzserver"
    "/opt/ros/noetic/lib/gazebo_ros/gzclient"
    # pgrep -f matches the COMMAND LINE, not the executable path.  gzclient's
    # argv[0] is plain "gzclient" even though the binary on disk is
    # /usr/bin/gzclient-11.15.1, so a pattern naming the versioned path matches
    # nothing and the GUI survives every reset -- carrying its CPU load into the
    # next run.  Basenames are what the command line actually contains.
    "gzclient"
    "gzserver -e ode"
    "robot_state_publisher"
    "controller_manager/spawner"
    "gazebo_ros/spawn_model"
    # The two planners.  Matched on the node script names, which is what the
    # command line holds -- not on the launch files.
    "cnav_node.py"
    "neupan"
    # The harness itself.  Without this a bench.py left over from a killed
    # roslaunch keeps publishing goals into the next run's fresh sim.
    "bench.py --planner"
)
for pat in "${PATTERNS[@]}"; do
    for pid in $(pgrep -f -- "$pat" 2>/dev/null); do
        case "$protect" in *" $pid "*) continue ;; esac
        kill -9 "$pid" 2>/dev/null
    done
done
sleep 4

survivors=$(pgrep -af "rosmaster|gzserver|gzclient|roslaunch|cnav_node|neupan|bench.py" 2>/dev/null)
kept=""
while read -r pid rest; do
    [ -z "$pid" ] && continue
    case "$protect" in *" $pid "*) continue ;; esac
    kept="$kept$pid $rest"$'\n'
done <<< "$survivors"

if [ -n "$kept" ]; then
    echo "STILL RUNNING:"
    printf '%s' "$kept"
    exit 1
fi
echo "clean"
