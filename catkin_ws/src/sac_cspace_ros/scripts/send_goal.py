#!/usr/bin/env python

"""Publish a single goal PoseStamped for sac_cspace_node.

  rosrun sac_cspace_ros send_goal.py 3.0 3.0
  rosrun sac_cspace_ros send_goal.py 3.0 3.0 --frame odom --topic /sac_cspace/goal

Coordinates are in the goal topic's frame (default `odom`, which is the frame the
policy reads its pose from unless ~map_frame / ~odom_topic say otherwise).
"""

import argparse
import sys

import rospy
from geometry_msgs.msg import PoseStamped


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("x", type=float, help="goal x, metres")
    parser.add_argument("y", type=float, help="goal y, metres")
    parser.add_argument("--frame", default="odom", help="header.frame_id")
    parser.add_argument("--topic", default="/sac_cspace/goal")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="seconds to wait before publishing")
    args = parser.parse_args(argv)

    rospy.init_node("sac_cspace_send_goal", anonymous=True)
    pub = rospy.Publisher(args.topic, PoseStamped, queue_size=1)

    rospy.sleep(args.delay)   # let the publisher connect to the subscriber

    msg = PoseStamped()
    msg.header.frame_id = args.frame
    msg.header.stamp = rospy.Time.now()
    msg.pose.position.x = args.x
    msg.pose.position.y = args.y
    msg.pose.orientation.w = 1.0
    pub.publish(msg)

    rospy.loginfo("sent goal (%.2f, %.2f) in frame '%s' on %s",
                  args.x, args.y, args.frame, args.topic)
    rospy.sleep(0.5)          # give the message time to actually leave


if __name__ == "__main__":
    main(rospy.myargv(argv=sys.argv)[1:])
