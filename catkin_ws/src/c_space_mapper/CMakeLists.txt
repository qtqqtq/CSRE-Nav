cmake_minimum_required(VERSION 2.8.3)
project(c_space_mapper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  nav_msgs
  std_msgs
  sensor_msgs
  cv_bridge
  c_space_mapper_msgs
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp nav_msgs std_msgs sensor_msgs cv_bridge c_space_mapper_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# ---- Core algorithm library ----
add_library(c_space_mapper_core
  src/configuration_mapper.cpp
  src/multi_footprint_collision_checker.cpp
  src/footprint_collision_checker.cpp
  src/costmap_2d.cpp
  src/costmap_math.cpp
  src/footprint.cpp
  src/array_parser.cpp
)

target_link_libraries(c_space_mapper_core
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# ---- mapper_node ----
add_executable(c_space_mapper_node
  src/mapper_main.cpp
  src/mapper_node.cpp
)
target_link_libraries(c_space_mapper_node c_space_mapper_core ${catkin_LIBRARIES})

# ---- lidar_to_occupancy_node ----
add_executable(lidar_to_occupancy_node
  src/lidar_to_occupancy_node.cpp
)
target_link_libraries(lidar_to_occupancy_node ${catkin_LIBRARIES})

# ---- Install ----
install(TARGETS c_space_mapper_node lidar_to_occupancy_node c_space_mapper_core
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(FILES
  config/mapper_node.yaml
  config/lidar_to_occupancy_node.yaml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# ---- Offline costray test (unit/) ----
catkin_install_python(PROGRAMS unit/fake_occupancy_pub.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY unit/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/unit
  FILES_MATCHING PATTERN "*.launch" PATTERN "*.rviz")
