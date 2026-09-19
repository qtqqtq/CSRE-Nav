cmake_minimum_required(VERSION 3.0.2)
project(cnav_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  visualization_msgs
  tf
)

catkin_package()

catkin_install_python(PROGRAMS
  src/cnav_node.py
  src/cnav_core.py
  scripts/send_goal.py
  scripts/check_models.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
