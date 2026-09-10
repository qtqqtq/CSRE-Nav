#!/usr/bin/env python3
"""Synthetic OccupancyGrid publisher for offline costray testing.

Publishes a robot-centric static map to /map_from_gazebo so that
c_space_mapper_node can generate cost rays without Gazebo or a real lidar.

The mapper_node computes cost rays from world (0, 0), so we build a 20 m x 20 m
grid with origin (-10, -10) and place a *symmetric* circular ring wall around
(0, 0) (inner radius 2.0 m, outer radius 2.4 m). Every one of the 36 beams then
hits the ring at roughly the same distance, which makes the /cost_ray_samples
fan look clean and symmetric. The small residual spread (a few cm) comes from
the robot's 0.44 m x 0.38 m rectangular footprint, which reaches slightly
farther along its diagonal than along its sides.
"""
import math

import rospy
from nav_msgs.msg import OccupancyGrid


def main():
    rospy.init_node("fake_occupancy_pub", anonymous=False)

    res = 0.05
    width = height = 400          # 20 m x 20 m
    origin_x = origin_y = -10.0   # world (0,0) -> cell (200, 200)

    # Symmetric ring wall centered on the robot (world origin).
    ring_inner = 2.0
    ring_outer = 2.4

    data = [0] * (width * height)
    for y in range(height):
        wy = origin_y + (y + 0.5) * res
        for x in range(width):
            wx = origin_x + (x + 0.5) * res
            r = math.hypot(wx, wy)
            if ring_inner <= r <= ring_outer:
                data[y * width + x] = 100   # LETHAL

    msg = OccupancyGrid()
    msg.header.frame_id = "base_link"
    msg.info.resolution = res
    msg.info.width = width
    msg.info.height = height
    msg.info.origin.position.x = origin_x
    msg.info.origin.position.y = origin_y
    msg.info.origin.orientation.w = 1.0
    msg.data = data

    pub = rospy.Publisher("/map_from_gazebo", OccupancyGrid, queue_size=1, latch=True)
    rate = rospy.Rate(1.0)
    rospy.loginfo("[fake_occupancy_pub] publishing %dx%d grid @ %.2f m, ring r=%.1f..%.1f m",
                  width, height, res, ring_inner, ring_outer)

    while not rospy.is_shutdown():
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    main()
