#!/bin/bash
# Build catkin workspace for DRL-robot-navigation
#
# Usage: bash /DRL-robot-navigation/build.bash

CATKIN_WS=/DRL-robot-navigation/catkin_ws

set -e

echo "=== Build catkin workspace ==="
cd $CATKIN_WS
source /opt/ros/noetic/setup.bash
catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5
source devel/setup.bash
echo "  catkin build done."

echo ""
echo "=== Build successful ==="
