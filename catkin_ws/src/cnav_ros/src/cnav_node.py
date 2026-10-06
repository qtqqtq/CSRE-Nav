#!/usr/bin/env python

"""Entry point for the CNN-SAC c-space navigation node.  See cnav_core."""

import os
import sys

import rospy

# the node is installed into lib/<pkg>/ next to its modules, and also runs in
# place from src/, so make the sibling import work in both cases
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cnav_core import CnavCore


if __name__ == "__main__":
    try:
        CnavCore().run()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        # Configuration problems (missing checkpoint, geometry mismatch, bad
        # backend) are fatal and must be loud rather than a silent traceback.
        rospy.logfatal("cnav: %s", exc)
        raise
