#!/usr/bin/env python

"""Entry point for the CNN-SAC c-space navigation node.  See sac_cspace_core."""

import os
import sys

import rospy

# the node is installed into lib/<pkg>/ next to its modules, and also runs in
# place from src/, so make the sibling import work in both cases
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sac_cspace_core import SacCspaceCore


if __name__ == "__main__":
    try:
        SacCspaceCore().run()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        # Configuration problems (missing checkpoint, geometry mismatch, bad
        # sac_dir) are fatal and must be loud rather than a silent traceback.
        rospy.logfatal("sac_cspace: %s", exc)
        raise
