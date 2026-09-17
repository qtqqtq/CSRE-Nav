cmake_minimum_required(VERSION 3.0.2)
project(planner_bench)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  std_srvs
  geometry_msgs
  nav_msgs
  sensor_msgs
  rosgraph_msgs
  gazebo_msgs
  tf
)

catkin_package()

# bench_common.py is installed alongside the scripts on purpose.  The scripts
# bootstrap their import path as `dirname(__file__)` plus `../src`, which covers
# both layouts: run from the source tree, bench_common is in ../src; installed,
# it is in this same lib/<pkg>/ directory.  Shipping it anywhere else breaks the
# installed case only, which is exactly the case that is never tested.
catkin_install_python(PROGRAMS
  scripts/gen_goals.py
  scripts/bench.py
  scripts/report.py
  src/bench_common.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
