cmake_minimum_required(VERSION 3.8)
project(c_space_mapper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
# find_package(nav2_costmap_2d REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(c_space_mapper_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

set(BEV_core
  src/multi_footprint_collision_checker.cpp
  src/configuration_mapper.cpp
  src/array_parser.cpp
  src/costmap_math.cpp
  src/costmap_2d.cpp
  src/footprint.cpp
  src/footprint_collision_checker.cpp
)

set(BEV_deps
  rclcpp
  geometry_msgs
  OpenCV
  c_space_mapper_msgs
  nav_msgs
  std_msgs
  sensor_msgs
)

add_executable(c_space_mapper_node
  src/mapper_node.cpp
  ${BEV_core}
)

target_include_directories(c_space_mapper_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(c_space_mapper_node
  ${BEV_deps}
)

add_executable(unit_test
  unit_test.cpp
  ${BEV_core}
)

ament_target_dependencies(unit_test
  ${BEV_deps}
)

target_include_directories(unit_test
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ---- LiDAR to Occupancy Grid converter node ----
add_executable(lidar_to_occupancy_node
  src/lidar_to_occupancy_node.cpp
)

ament_target_dependencies(lidar_to_occupancy_node
  rclcpp
  sensor_msgs
  nav_msgs
  std_msgs
)

target_include_directories(lidar_to_occupancy_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ---- Install ----
install(TARGETS c_space_mapper_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS unit_test
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS lidar_to_occupancy_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  config/lidar_to_occupancy_node.yaml
  config/mapper_node.yaml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
