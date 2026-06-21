"""Launch lidar_to_occupancy + c_space_mapper + ros1_bridge.

Prerequisites (run in shell before this):
  export ROS_MASTER_URI=http://localhost:11311
  source /opt/ros/noetic/setup.bash
  source /home/BEVNav/cakin_ws/devel/setup.bash
  source /opt/ros/foxy/setup.bash
  source /home/BEVNav/colcon_ws/install/setup.bash

Then:
  ros2 launch c_space_mapper c_space_mapper.launch.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("c_space_mapper")

    lidar_cfg = LaunchConfiguration(
        "lidar_config",
        default=os.path.join(pkg_share, "lidar_to_occupancy_node.yaml"))
    mapper_cfg = LaunchConfiguration(
        "mapper_config",
        default=os.path.join(pkg_share, "mapper_node.yaml"))

    ros1_setup = "/opt/ros/noetic/setup.bash"
    ros2_setup = "/opt/ros/foxy/setup.bash"
    bridge_cmd = [
        "bash", "-c",
        f"source {ros1_setup} && source {ros2_setup} && "
        "ros2 run ros1_bridge dynamic_bridge --bridge-all-topics"
    ]

    return LaunchDescription([
        DeclareLaunchArgument("lidar_config", default_value=lidar_cfg),
        DeclareLaunchArgument("mapper_config", default_value=mapper_cfg),

        LogInfo(msg="=== c_space_mapper launch ==="),
        LogInfo(msg=["lidar config: ", lidar_cfg]),
        LogInfo(msg=["mapper config: ", mapper_cfg]),

        # 先启动 lidar_to_occupancy，采集几帧点云后再启 mapper
        Node(package="c_space_mapper", executable="lidar_to_occupancy_node",
             name="lidar_to_occupancy_node", output="screen",
             parameters=[lidar_cfg]),

        Node(package="c_space_mapper", executable="c_space_mapper_node",
             name="c_space_mapper_node", output="screen",
             parameters=[mapper_cfg]),

        ExecuteProcess(cmd=bridge_cmd, output="screen", name="ros1_bridge"),
    ])
