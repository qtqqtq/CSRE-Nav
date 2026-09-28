#!/usr/bin/env bash
# Load the benchmark profile as private ROS parameters, then run one planner.
set -euo pipefail
BENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$BENCH_DIR/.." && pwd)
case "${1:-}" in
    cspace)
        rosparam load "$BENCH_DIR/config/cnav_limo.yaml" /cnav_node
        rosparam set /cnav_node/sac_dir "${PLANNER_BENCH_SAC_DIR:-$REPO_DIR/SAC}"
        exec rosrun cnav_ros cnav_node.py __name:=cnav_node
        ;;
    neupan)
        rosparam set /neupan_control/config_file "$BENCH_DIR/config/neupan_limo_bench.yaml"
        rosparam set /neupan_control/map_frame odom
        rosparam set /neupan_control/base_frame base_link
        rosparam set /neupan_control/lidar_frame laser_link
        rosparam set /neupan_control/scan_range '0.0 5.0'
        rosparam set /neupan_control/scan_angle_range '-3.14 3.14'
        rosparam set /neupan_control/marker_size 0.05
        rosparam set /neupan_control/marker_z 1.0
        rosparam set /neupan_control/scan_downsample 6
        rosparam set /neupan_control/refresh_initial_path false
        rosparam set /neupan_control/dune_checkpoint \
            "${PLANNER_BENCH_NEUPAN_CKPT:-$(rospack find neupan_ros)/example/gazebo_limo/pretrain_limo/model_5000.pth}"
        exec rosrun neupan_ros neupan_node.py __name:=neupan_control \
            /scan:=/limo/scan /neupan_cmd_vel:=/cmd_vel \
            /neupan_goal:=/move_base_simple/goal
        ;;
    *) echo "usage: $0 cspace|neupan" >&2; exit 2 ;;
esac
