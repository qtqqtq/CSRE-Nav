cmake_minimum_required(VERSION 2.8.3)
project(robot_2dnav)

find_package(catkin REQUIRED)

catkin_package()

# Install configs and launch files
install(DIRECTORY launch/ maps/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY ./
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  FILES_MATCHING PATTERN "*.yaml"
)
