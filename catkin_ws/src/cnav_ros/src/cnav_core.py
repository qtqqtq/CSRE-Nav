#!/usr/bin/env python

"""
cnav_core -- ROS 1 wrapper for the CNN-SAC c-space (NavRL) navigation policy.

The policy is the cost-ray CNN actor trained on `Sparrow_V2.SparrowCspaceEnv`.
Its observation is

    state = flatten(normalize(ray_segments))
            + [distance, heading, normalized_actual_v, normalized_actual_w]

where the cost rays are built from the robot's line-of-sight lidar hits, exactly
as `SparrowCspaceEnv` does.  This node rebuilds that observation from live ROS
sensor data with `cnav_cspace`, the vendored copy of the training tree's
`cspace_cost_rays.py` that ships in this package (`src/cnav_cspace.py`), using
live laser scans and odometry.  Nothing here reaches into a SAC checkout: the
package is self-contained, with its checkpoints under `model/`.

The kinematic block contains goal-relative geometry and normalized velocity from
odometry.  These are measured velocities, not the previous actor command, matching
the post-filter `car_state[:, 3:5]` values used during Sparrow training.

THREE things differ between checkpoint families, and each is a knob here:

  1. `architecture` -- sniffed out of the state dict, never configured: backbone
     4->8->8 stride 2 vs 4->16->16 stride [1,1]/[1,2], and head
     `layer_3 -> action_dim*2` (stochastic, tanh of the mean) vs
     `layer_3 -> action_dim` (deterministic, tanh inside forward).
  2. `geometry` -- nz / a / cost_ray_max_range / footprint.  Not
     recoverable from a checkpoint, so it comes from the YAML and is *validated*
     against the loaded network's fc width; a mismatch aborts the node instead of
     silently degrading the policy.
  3. `ray_frame` + `preprocess` + velocity scale -- the Sparrow-trained models read
     a robot-centric grid whose 2nd axis runs to the robot's RIGHT
     (`_build_occ_grid` uses `rel_y = dx*sin(t) + dy*cos(t)`, the mirror of the
     standard ROS body frame) with configurable normalized distance channels and a 0.5 m/s /
     2 rad/s action scale; the older Gazebo models read ROS-frame rays from the C++
     mapper with a linear `max_range - d` encoding and the actor output published
     to /cmd_vel unscaled.

The current deployment profile is `config/cnav_sparrow.yaml` for the
one-channel, 36x30 first-entry model `exact_castray_interval1_0`. Legacy profiles
are retained as geometry references but are intentionally rejected by this build.

Developed for the DRL-robot-navigation workspace.
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rospy
import tf
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image, LaserScan
from visualization_msgs.msg import Marker

# The node runs both from src/ and from lib/<pkg>/ after catkin_install_python,
# so make the sibling module importable before touching it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cnav_cspace import (footprint_constraints, pointset_batch,
                         preprocess_cost_ray_intervals)

# The Limo's URDF collision union (0.29 x 0.22 m: its 0.13 x 0.12 m chassis box
# plus the four wheel cylinders), uninflated, in `base_frame`.  See `~body`.
BODY_POLYGON_DEFAULT = [[0.145, 0.11], [0.145, -0.11], [-0.145, -0.11], [-0.145, 0.11]]

# The same union plus a 2 cm observation margin.  See `~footprint`.
FOOTPRINT_POLYGON_DEFAULT = [[0.165, 0.13], [0.165, -0.13],
                             [-0.165, -0.13], [-0.165, 0.13]]


# ---------------------------------------------------------------------------
# Vendored actor networks
#
# The two generations live in different git revisions of `SAC/cnn_network.py`, so
# they are reproduced here rather than imported: today's file holds only the new
# backbone, and the old one is not on disk at all.  Both are faithful copies of
# `StaticObstacleCNN` / `CNNActor` as of the revision that trained the
# corresponding checkpoints (new: 7f6982c, old: 289333b).  Note the old actor
# applies tanh inside `forward` and its state dict carries a bare `tanh` key.
# ---------------------------------------------------------------------------
class _BackboneNew(nn.Module):
    """StaticObstacleCNN, post-refactor: both axes halved twice (1->4->8->8)."""

    def __init__(self, in_channels=1, h_beams=36, v_layers=30, feature_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=[5, 3], padding=[2, 1])
        self.conv2 = nn.Conv2d(4, 8, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1])
        self.conv3 = nn.Conv2d(8, 8, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1])

        self.fc = nn.Linear(self._flat_size(in_channels, h_beams, v_layers), feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def _flat_size(self, c, h, v):
        with torch.no_grad():
            d = self.conv3(self.conv2(self.conv1(torch.zeros(1, c, h, v))))
        return d.numel() // d.shape[0]

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = self.fc(x.flatten(start_dim=1))
        return self.norm(F.elu(x))


class _BackboneOld(nn.Module):
    """StaticObstacleCNN, pre-refactor: 1->4->16->16, stride [1,1] then [1,2]."""

    def __init__(self, in_channels=1, h_beams=36, v_layers=10, feature_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=[5, 3], padding=[2, 1])
        self.conv2 = nn.Conv2d(4, 16, kernel_size=[5, 3], stride=[1, 1], padding=[2, 1])
        self.conv3 = nn.Conv2d(16, 16, kernel_size=[5, 3], stride=[1, 2], padding=[2, 1])

        self.fc = nn.Linear(self._flat_size(in_channels, h_beams, v_layers), feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def _flat_size(self, c, h, v):
        with torch.no_grad():
            d = self.conv3(self.conv2(self.conv1(torch.zeros(1, c, h, v))))
        return d.numel() // d.shape[0]

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = self.fc(x.flatten(start_dim=1))
        return self.norm(F.elu(x))


class _ActorBase(nn.Module):
    """ray_grid + kinematic -> MLP[400 -> 300] trunk.  Subclasses differ only in
    the output layer (action vs mean/log_std)."""

    def __init__(self, backbone, feature_dim, kinematic_dim):
        super().__init__()
        self.cnn = backbone
        self.layer_1 = nn.Linear(feature_dim + kinematic_dim, 400)
        self.layer_2 = nn.Linear(400, 300)

    def _trunk(self, ray_grid, kinematic):
        x = torch.cat([self.cnn(ray_grid), kinematic], dim=-1)
        return F.leaky_relu(self.layer_2(F.leaky_relu(self.layer_1(x))))


class _ActorStochastic(_ActorBase):
    """SAC actor: `layer_3` emits mean and log_std; eval uses tanh(mean)."""

    def __init__(self, backbone, feature_dim, kinematic_dim, action_dim):
        super().__init__(backbone, feature_dim, kinematic_dim)
        self.layer_3 = nn.Linear(300, action_dim * 2)

    def get_mean_action(self, ray_grid, kinematic):
        mean = self.layer_3(self._trunk(ray_grid, kinematic)).chunk(2, dim=-1)[0]
        return torch.tanh(mean)

    def sample(self, ray_grid, kinematic):
        mean, log_std = self.layer_3(self._trunk(ray_grid, kinematic)).chunk(2, dim=-1)
        std = torch.clamp(log_std, -20.0, 2.0).exp()
        return torch.tanh(mean + std * torch.randn_like(std))


class _ActorDeterministic(_ActorBase):
    """Pre-SAC actor: `layer_3` -> action, tanh applied inside forward."""

    def __init__(self, backbone, feature_dim, kinematic_dim, action_dim):
        super().__init__(backbone, feature_dim, kinematic_dim)
        self.layer_3 = nn.Linear(300, action_dim)

    def get_mean_action(self, ray_grid, kinematic):
        return torch.tanh(self.layer_3(self._trunk(ray_grid, kinematic)))

    sample = get_mean_action


def _arch_of(state_dict):
    """Sniff (backbone, head, feature_dim) out of an actor state dict.

    Raises ValueError with a readable message when the checkpoint matches neither
    known generation (e.g. the MLP `SAC_laser_*` actors, which this node does not
    support).
    """
    missing = [k for k in ("cnn.conv2.weight", "cnn.norm.weight", "layer_3.weight")
               if k not in state_dict]
    if missing:
        raise ValueError(
            "checkpoint does not look like a CNN c-space actor (missing %s). "
            "The MLP variants (SAC_laser_*) are not supported by this node."
            % ", ".join(missing))

    conv2_out = state_dict["cnn.conv2.weight"].shape[0]
    feature_dim = state_dict["cnn.norm.weight"].shape[0]
    layer3_out = state_dict["layer_3.weight"].shape[0]

    if conv2_out == 8:
        backbone = "new"
    elif conv2_out == 16:
        backbone = "old"
    else:
        raise ValueError("unknown backbone: cnn.conv2 has %d output channels "
                         "(expected 8 or 16)" % conv2_out)

    # action_dim is 2 for every c-space model; the stochastic head doubles it.
    # All 11 shipped cspace checkpoints are stochastic (verified); the
    # deterministic branch covers the `CNNActor` variant, which has no saved model.
    head = "deterministic" if layer3_out == 2 else "stochastic"
    return backbone, head, feature_dim


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# Default observation layout for the current cnav_sparrow.yaml profile.
RAY_CHANNELS = 1
KINEMATIC_DIM = 4

# Where this package keeps the checkpoints it ships with.
DEFAULT_MODEL_DIR = "model"


def package_dir():
    """Root of the cnav_ros package, from either the source tree or an install space.

    Running from the source tree the file sits in `<pkg>/src/`, so the package
    root is its parent and `model/` is right there.  `catkin_install_python`
    instead puts it in `lib/<pkg>/`, where that parent is `lib/`; ask ros for the
    package path then.  (In a devel-space build rospack also resolves to the
    source tree, so both branches agree.)
    """
    here = os.path.dirname(os.path.abspath(__file__))
    local = os.path.normpath(os.path.join(here, os.pardir))
    if os.path.isdir(os.path.join(local, DEFAULT_MODEL_DIR)):
        return local
    try:
        import rospkg
        return rospkg.RosPack().get_path("cnav_ros")
    except Exception:                  # no ros / package not on ROS_PACKAGE_PATH
        return local


def resolve_model_dir(model_dir):
    """Absolute checkpoint directory.  Relative paths resolve under the package.

    Models travel with the package, so `model` -- the default -- always means
    this package's own `model/` directory, whatever the working directory is.
    """
    if not model_dir:
        model_dir = DEFAULT_MODEL_DIR
    if os.path.isabs(model_dir):
        return model_dir
    return os.path.join(package_dir(), model_dir)


def points_in_polygon(px, py, polygon):
    """Even-odd point-in-polygon, vectorised over the points.

    `polygon` is a closed list of (x, y) vertices in the same frame as the
    points; the edge from the last vertex back to the first is included, so the
    winding direction does not matter.

    This is the body test in `~body`, and it is deliberately a containment test
    rather than a ray-exit distance: the Limo's laser sits *outside* the shape it
    clips (the chassis box is 3.8 cm astern of it), so "how far to the polygon's
    far edge along this bearing" is undefined for most bearings while "is this
    hit point on the robot" is well posed everywhere.
    """
    inside = np.zeros(np.shape(px), dtype=bool)
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        # Half-open in y, so a vertex exactly at a point's height counts once.
        straddle = (y0 > py) != (y1 > py)
        dy = y1 - y0
        # Only straddling edges can contribute, and they have dy != 0, so the
        # guarded denominator never feeds a live branch.
        x_cross = x0 + (py - y0) * (x1 - x0) / np.where(dy == 0.0, 1.0, dy)
        inside ^= straddle & (px < x_cross)
    return inside


def _kin_dim_of(state_dict, feature_dim, name=None):
    """Sniff the kinematic-block width out of an actor state dict.

    The block is concatenated with the CNN features and goes into `layer_1`, so
    that layer's input width is the checkpoint's only remaining trace of it.
    The current four values are `[distance, heading, normalized_actual_v,
    normalized_actual_w]`.  A matching width alone does not make older
    previous-command checkpoints semantically compatible; their one-channel
    convolution is rejected separately by `_ray_channels_of`.

    Returns the width, or raises ValueError with a readable message.  Shared by
    the node and check_models.py so the two cannot disagree about the layout.
    """
    kin_dim = state_dict["layer_1.weight"].shape[1] - feature_dim
    if kin_dim != KINEMATIC_DIM:
        raise ValueError(
            "%swas trained with a %d-dim kinematic block (layer_1 expects "
            "feature_dim+%d), but this build uses %d ([distance, heading, "
            "normalized_actual_v, normalized_actual_w]). Retrain the model "
            "with the current observation layout."
            % ("checkpoint '%s' " % name if name else "", kin_dim, kin_dim,
               KINEMATIC_DIM))
    return kin_dim


def _ray_channels_of(state_dict, expected_channels, name=None):
    """Validate the configured ray channel count against the first conv."""
    ray_channels = state_dict["cnn.conv1.weight"].shape[1]
    if ray_channels != expected_channels:
        raise ValueError(
            "%swas trained with %d ray channel(s), but this build uses %d "
            "from the configured interval count. Select the matching model "
            "and cost_ray_num_intervals setting."
            % ("checkpoint '%s' " % name if name else "", ray_channels,
               expected_channels))
    return ray_channels


class CnavCore(object):
    """Subscribes to a laser scan + odometry + goal, publishes velocity commands."""

    def __init__(self):
        rospy.init_node("cnav_node", anonymous=True)

        # ---- model ------------------------------------------------------
        self.model_name = rospy.get_param(
            "~model_name", "exact_castray_interval1_0")
        self.model_dir = resolve_model_dir(rospy.get_param("~model_dir", ""))
        self.device = torch.device(rospy.get_param("~device", "cuda"))
        self.stochastic = bool(rospy.get_param("~stochastic", False))

        # ---- frames / topics --------------------------------------------
        self.map_frame = rospy.get_param("~map_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.lidar_frame = rospy.get_param("~lidar_frame", "laser_link")
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.cmd_topic = rospy.get_param("~cmd_vel_topic", "/cnav_cmd_vel")
        self.goal_topic = rospy.get_param("~goal_topic", "/cnav_goal")
        self.plan_topic = rospy.get_param("~plan_topic", "/cnav_cost_ray_path")
        self.marker_topic = rospy.get_param("~marker_topic", "/cnav_goal_marker")
        self.cost_ray_image_topic = rospy.get_param("~cost_ray_image_topic",
                                                    "/cnav_cost_ray_image")

        # 10 Hz matches the training control interval (Sparrow ctrl_interval = 0.1 s)
        self.rate_hz = float(rospy.get_param("~rate", 10.0))
        self.goal_tolerance = float(rospy.get_param("~goal_tolerance", 0.3))
        self.publish_debug = bool(rospy.get_param("~publish_debug", True))

        # Pixels per cost-ray cell in the debug image; 5 is what play_cspace.py
        # draws, giving 36*5 x nz*5 = 180 x 150.  Bump it on a large screen --
        # rviz renders the image 1:1 and does not scale it.
        self.cost_ray_image_cell = int(rospy.get_param("~cost_ray_image_cell", 5))
        if self.cost_ray_image_cell < 1:
            raise ValueError("~cost_ray_image_cell must be >= 1, got %d"
                             % self.cost_ray_image_cell)

        # ---- geometry (must match the checkpoint's training run) ---------
        self.nz = int(rospy.get_param("~nz", 30))
        self.a = float(rospy.get_param("~a", 2.0 * math.pi / 30.0))
        self.max_range = float(rospy.get_param("~cost_ray_max_range", 3.0))
        self.num_beams = int(rospy.get_param("~cost_ray_num_beams", 36))
        self.h_res_deg = float(rospy.get_param("~cost_ray_h_res_degrees", 10.0))
        self.num_intervals = int(rospy.get_param(
            "~cost_ray_num_intervals", RAY_CHANNELS))
        if self.num_intervals < 1:
            raise ValueError("~cost_ray_num_intervals must be >= 1")
        self.footprint = [[float(x), float(y)] for x, y in rospy.get_param(
            "~footprint", FOOTPRINT_POLYGON_DEFAULT)]

        # `env`: the grid's 2nd axis runs to the robot's RIGHT, as built by
        #        SparrowCspaceEnv._build_occ_grid (both Sparrow families).
        # `ros`: standard ROS body frame (2nd axis = left); what the C++ c-space
        #        mapper fed the Gazebo-trained models.
        self.ray_frame = rospy.get_param("~ray_frame", "env")
        if self.ray_frame not in ("env", "ros"):
            raise ValueError("~ray_frame must be 'env' or 'ros', got %r" % self.ray_frame)
        self.mirror_y = (self.ray_frame == "env")

        # `interval`: metric radial segments, each divided by max_range.
        self.preprocess = rospy.get_param("~preprocess", "interval")
        if self.preprocess != "interval":
            raise ValueError("~preprocess must be 'interval' for the current "
                             "observation, got %r" % self.preprocess)

        # ---- action scaling ---------------------------------------------
        # Sparrow: v_phys = continuous_scale * a with (50 cm/s, 2 rad/s), a0 in
        # [-1,1] taken as-is (negative = reverse).  The Gazebo pipeline published
        # the actor output straight to /cmd_vel, so its scale is 1.0 in m/s and rad/s.
        self.v_linear_max = float(rospy.get_param("~v_linear_max", 0.5))
        self.v_angular_max = float(rospy.get_param("~v_angular_max", 2.0))
        if self.v_linear_max <= 0.0 or self.v_angular_max <= 0.0:
            raise ValueError("velocity limits must be positive, got (%.3f, %.3f)"
                             % (self.v_linear_max, self.v_angular_max))

        # Match Sparrow's first-order actuator model:
        #   velocity = K * previous_velocity + (1 - K) * target_velocity
        # A zero K keeps the legacy direct-output behaviour.
        filter_k = rospy.get_param("~velocity_filter_k", [0.0, 0.0])
        if not isinstance(filter_k, (list, tuple)) or len(filter_k) != 2:
            raise ValueError("~velocity_filter_k must be [linear_k, angular_k]")
        self.velocity_filter_k = np.asarray(filter_k, dtype=np.float64)
        if (not np.all(np.isfinite(self.velocity_filter_k))
                or np.any(self.velocity_filter_k < 0.0)
                or np.any(self.velocity_filter_k >= 1.0)):
            raise ValueError("~velocity_filter_k values must be finite and in [0, 1)")
        self.filtered_velocity = np.zeros(2, dtype=np.float64)

        # Hits at or beyond this range count as "no return", mirroring the sim's
        # `ld_scan_result < ld_d_range`.  0 => use cost_ray_max_range.
        scan_max = float(rospy.get_param("~scan_max_range", 0.0))
        self.scan_max_range = scan_max if scan_max > 0.0 else self.max_range
        self.scan_min_range = float(rospy.get_param("~scan_min_range", 0.0))

        # The robot's own chassis, as a polygon in `base_frame` (+x forward, +y
        # left, metres).  Hits inside it are the laser seeing the robot and are
        # dropped before the cost rays are built; `[]` disables the test.
        #
        # Default: the Limo's URDF collision union -- its 0.13 x 0.12 m chassis
        # box plus the four wheel cylinders, i.e. 0.29 x 0.22 m -- left
        # uninflated, so that it stays strictly inside the default `~footprint`
        # (the same union + 2 cm).  That nesting is load-bearing, not tidy:
        # `entry = 0` iff a hit is inside the *footprint*, so a body that covered
        # the whole footprint would hide every genuine contact and the
        # observation would lose its collision signal.  The visible contact band
        # is the 2 cm shell between the two polygons.
        #
        # A scalar range cutoff cannot do this job.  Measured on the Limo in an
        # empty world, the self-return appears only while driving forward (not at
        # rest, not spinning, not reversing) and only on the last two beams at
        # each end of the FOV, at 0.0500-0.0931 m -- so the largest useful scalar
        # cutoff (0.09) is *below* the worst self-return, while a cutoff large
        # enough to cover it would also discard every genuine return out to 9 cm
        # on every other bearing.
        body = rospy.get_param("~body", BODY_POLYGON_DEFAULT)
        if body in (None, [], ()):
            self.body = None
        else:
            verts = np.asarray(body, dtype=np.float64)
            if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 3:
                raise ValueError("~body must be a list of >= 3 [x, y] pairs, got %r"
                                 % (body,))
            if not np.all(np.isfinite(verts)):
                raise ValueError("~body has a non-finite vertex: %r" % (body,))
            self.body = [tuple(v) for v in verts]

        self._load_actor()
        self._build_geometry()

        # ---- state ------------------------------------------------------
        self.robot_state = None        # np.array([x, y, yaw]) in `map_frame`
        self.robot_velocity = np.zeros(2, dtype=np.float64)  # [m/s, rad/s]
        self.goal = None               # np.array([x, y])
        self.hit_points = None         # (M, 2) lidar hits, c-space frame, metres
        self.n_body_hits = 0           # returns dropped by the `~body` test
        self.arrived = False

        self.vel_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.plan_pub = rospy.Publisher(self.plan_topic, Path, queue_size=1)
        self.marker_pub = rospy.Publisher(self.marker_topic, Marker, queue_size=1)
        self.cost_ray_image_pub = rospy.Publisher(self.cost_ray_image_topic, Image,
                                                  queue_size=1)

        self.listener = tf.TransformListener()
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber(self.goal_topic, PoseStamped, self.goal_callback, queue_size=1)

        rospy.loginfo("cnav: model=%s backbone=%s head=%s feature_dim=%d device=%s",
                      self.model_name, self.backbone, self.action_head,
                      self.feature_dim, self.device)
        rospy.loginfo("cnav: geometry nz=%d a=%.4f range=%.2f beams=%d -> "
                      "ray_grid=%dx%d, state_dim=%d",
                      self.nz, self.a, self.max_range,
                      self.num_beams, self.num_beams, self.nz,
                      self.num_intervals * self.num_beams * self.nz + KINEMATIC_DIM)
        rospy.loginfo("cnav: ray_frame=%s preprocess=%s scale=(%.2f m/s, %.2f rad/s) "
                      "filter_k=(%.2f, %.2f) rate=%.1f Hz",
                      self.ray_frame, self.preprocess,
                      self.v_linear_max, self.v_angular_max,
                      self.velocity_filter_k[0], self.velocity_filter_k[1],
                      self.rate_hz)
        rospy.loginfo("cnav: waiting for a goal on %s", self.goal_topic)

    # ------------------------------------------------------------------ setup
    def _load_actor(self):
        """Load <model_name>_actor.pth, building a network to match its shapes."""
        path = os.path.join(self.model_dir, "%s_actor.pth" % self.model_name)
        if not os.path.exists(path):
            raise IOError("actor checkpoint not found: %s" % path)

        state = torch.load(path, map_location="cpu", weights_only=True)
        self.backbone, self.action_head, self.feature_dim = _arch_of(state)

        backbone_cls = _BackboneNew if self.backbone == "new" else _BackboneOld
        actor_cls = (_ActorStochastic if self.action_head == "stochastic"
                     else _ActorDeterministic)
        actor = actor_cls(
            backbone_cls(in_channels=self.num_intervals, h_beams=self.num_beams,
                         v_layers=self.nz,
                         feature_dim=self.feature_dim),
            self.feature_dim, KINEMATIC_DIM, 2)

        _ray_channels_of(state, self.num_intervals, self.model_name)
        _kin_dim_of(state, self.feature_dim, self.model_name)

        # The ray grid comes from the config; the checkpoint pins only the fc
        # width, which is a product of both axes.  Compare *those* first so the
        # failure names the parameter at fault instead of dumping a state_dict diff.
        want = state["cnn.fc.weight"].shape[1]
        got = actor.cnn.fc.weight.shape[1]
        if want != got:
            raise RuntimeError(
                "c-space geometry mismatch for '%s': the checkpoint expects a CNN "
                "flatten of %d, but nz=%d with %d beams gives %d. The observation "
                "would be built at the wrong theta resolution and the policy would "
                "silently degrade -- pick the config matching this checkpoint's "
                "generation (see config/*.yaml)."
                % (self.model_name, want, self.nz, self.num_beams, got))

        actor.load_state_dict(state)
        self.actor = actor.to(self.device).eval()

    def _build_geometry(self):
        """One-time footprint geometry: the outward edge normals, edge offsets
        and per-(beam, theta) cosines of the fixed footprint.

        The offsets are rotation-invariant (`h` does not depend on theta), so
        the whole table is built once, here, and merely broadcast over every
        scan in `_pointset_intervals`.
        """
        self._fp_nrm, self._fp_h, self._fp_c = footprint_constraints(
            [np.asarray(self.footprint, dtype=np.float64)],
            num_beams=self.num_beams, nz=self.nz, a=self.a,
            h_res_degrees=self.h_res_deg, device=self.device)

    # ------------------------------------------------------------- callbacks
    def odom_callback(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.z * q.z + q.y * q.y))
        self.robot_state = np.array([p.x, p.y, yaw], dtype=np.float64)
        self.robot_velocity[:] = (
            msg.twist.twist.linear.x,
            msg.twist.twist.angular.z,
        )

    def goal_callback(self, msg):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y],
                             dtype=np.float64)
        self.arrived = False
        rospy.loginfo("cnav: new goal (%.2f, %.2f)", self.goal[0], self.goal[1])
        if self.publish_debug:
            self._publish_goal_marker()

    def scan_callback(self, msg):
        """Turn one scan into the obstacle point set the cost rays are built from.

        The obstacle model is the lidar hit endpoints themselves, unquantised:
        there is no raster, so an obstacle keeps full metric precision all the
        way into the contact test and two adjacent hits cannot round *past* each
        other into a phantom gap.
        """
        if self.robot_state is None:
            return

        ranges = np.asarray(msg.ranges, dtype=np.float64)
        if ranges.size == 0:
            return
        angles = msg.angle_min + np.arange(ranges.size) * msg.angle_increment

        lo = max(msg.range_min, self.scan_min_range)
        hi = min(msg.range_max, self.scan_max_range)
        usable = (~np.isnan(ranges)) & (ranges > lo)
        if not np.any(usable):
            self.hit_points = None
            return

        r_all, th_all = np.minimum(ranges[usable], hi), angles[usable]
        hit = np.isfinite(r_all) & (r_all < hi)
        r, th = r_all[hit], th_all[hit]

        # lidar -> base_link (this is what carries the robot's heading)
        try:
            (trans, rot) = self.listener.lookupTransform(
                self.base_frame, self.lidar_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException):
            rospy.loginfo_throttle(2.0, "cnav: waiting for tf %s -> %s",
                                   self.base_frame, self.lidar_frame)
            return

        # quaternion is (x, y, z, w)
        lyaw = math.atan2(2.0 * (rot[3] * rot[2] + rot[0] * rot[1]),
                          1.0 - 2.0 * (rot[2] * rot[2] + rot[1] * rot[1]))

        px, py = r * np.cos(th), r * np.sin(th)
        cy, sy = math.cos(lyaw), math.sin(lyaw)
        # body frame, metres: `rel_x` forward, `rel_y` to the robot's LEFT
        rel_x = (px * cy - py * sy) + trans[0]
        rel_y = (px * sy + py * cy) + trans[1]

        # Drop the robot's own chassis, in `base_frame` and before the c-space
        # frame flip, so the polygon stays a plain robot-geometry description.
        # See `~body` for why this is a containment test and not a range cutoff.
        self.n_body_hits = 0
        if self.body is not None:
            keep = ~points_in_polygon(rel_x, rel_y, self.body)
            self.n_body_hits = int(np.count_nonzero(~keep))
            if self.n_body_hits:
                rel_x, rel_y = rel_x[keep], rel_y[keep]

        # The trained grid's axis 1 runs to the robot's RIGHT, so the ROS left
        # component is negated (see `ray_frame`).
        j = -rel_y if self.mirror_y else rel_y

        # Obstacle points in the cost-ray frame, metres.  A beam that returned
        # nothing is simply absent, which is equivalent to the training env's
        # "push it far outside the window": the nearest contact with such a
        # point is ~10x the sensor range away.
        self.hit_points = np.stack((rel_x, j), axis=1)

    # ------------------------------------------------------------- inference
    def _pointset_intervals(self):
        """Analytic (entry, width) ray segments from the lidar hit points.

        Mirrors `SparrowCspaceEnv._pointset_intervals`: channel 0 is the entry
        distance and channel 1 the contact's own width, in metres, shaped
        `(1, C, B, nz)`.  Only the configured channels are kept; a C above 2 is
        degenerate once obstacles are points (there is no second gap), so the
        extra ones are zeroed rather than left undefined.
        """
        shape = (1, self.num_beams, self.nz)
        entry = torch.full(shape, self.max_range, dtype=torch.float32,
                           device=self.device)
        width = torch.zeros(shape, dtype=torch.float32, device=self.device)
        if self.hit_points is not None and self.hit_points.shape[0] > 0:
            pts = torch.as_tensor(self.hit_points[None, ...],
                                  dtype=torch.float32, device=self.device)
            entry, width, _collide = pointset_batch(
                pts, self._fp_nrm, self._fp_h, self._fp_c,
                self.a, self.max_range)

        channels = [entry, width][:self.num_intervals]
        while len(channels) < self.num_intervals:
            channels.append(torch.zeros_like(entry))
        return torch.stack(channels, dim=1)                     # (1,C,B,nz) m

    def _build_perception(self):
        """Scan -> ray segments -> processed ray grid.

        Returns `(entry_rays, cr4)`: raw first-collision distances `(1,B,nz)`
        in metres and the processed `(1,C,B,nz)` ray grid consumed by the
        CNN.  The raw entry channel is retained for the RViz polar profile.

        Deliberately goal-free, which is what lets `run()` publish the debug
        topics before any goal has been set.  A viewer opened before the first
        click would otherwise sit on "No Image" -- the topics are advertised
        from startup, so it looks like a dead publisher rather than a policy
        that is simply not driving anywhere yet.
        """
        intervals = self._pointset_intervals()
        cr4 = preprocess_cost_ray_intervals(
            intervals, max_range=self.max_range).float()        # (1,C,B,nz)
        return intervals[:, 0], cr4

    def _build_state(self, rays, cr4):
        """Assemble the (1, C*B*nz + KINEMATIC_DIM) observation.

        Takes the perception half as an argument rather than recomputing it, so
        that `run()` can build it once and still publish it on the ticks where
        no goal exists.  Returns `(state, distance)`.
        """
        cr = cr4.reshape(1, -1)

        dx = self.goal[0] - self.robot_state[0]
        dy = self.goal[1] - self.robot_state[1]
        distance = math.hypot(dx, dy)
        # Standard ROS bearing error; numerically identical to the env's
        # atan2(robot - goal) + pi/2 - yaw formulation.
        heading = wrap_to_pi(math.atan2(dy, dx) - self.robot_state[2])

        normalized_velocity = np.clip(
            self.robot_velocity
            / np.array([self.v_linear_max, self.v_angular_max]),
            -1.0, 1.0)
        kin = torch.tensor([[distance, heading,
                             normalized_velocity[0], normalized_velocity[1]]],
                           dtype=torch.float32, device=self.device)
        return torch.cat([cr, kin], dim=1), distance

    def _command_to_twist(self, action):
        """Actor output in [-1,1] -> filtered physical Twist (m/s, rad/s).

        The actor's linear output is already the signed velocity fraction
        (negative = reverse), so it scales straight through -- there is no
        forward-only remapping.
        """
        target_velocity = np.array([
            self.v_linear_max * float(action[0]),
            self.v_angular_max * float(action[1]),
        ], dtype=np.float64)
        self.filtered_velocity = (
            self.velocity_filter_k * self.filtered_velocity
            + (1.0 - self.velocity_filter_k) * target_velocity
        )

        twist = Twist()
        twist.linear.x = self.filtered_velocity[0]
        twist.angular.z = self.filtered_velocity[1]
        return twist

    def _reset_command_filter(self):
        self.filtered_velocity.fill(0.0)

    def run(self):
        rate = rospy.Rate(self.rate_hz)

        while not rospy.is_shutdown():
            if self.robot_state is None:
                rate.sleep()
                continue

            rays, cr4 = self._build_perception()

            # Published before the goal check on purpose, in both directions:
            # a stopped robot is exactly when "what does the policy perceive
            # right now" is worth looking at, and a robot with no goal at all
            # still has perception worth showing.  Both topics used to be
            # silent until the first goal and again once it was reached.
            if self.publish_debug:
                self._publish_cost_ray_path(rays)
                self._publish_cost_ray_image(cr4)

            if self.goal is None:
                rospy.logwarn_throttle(
                    5.0, "cnav: no goal yet -- publish a PoseStamped on %s",
                    self.goal_topic)
                rate.sleep()
                continue

            state, distance = self._build_state(rays, cr4)

            with torch.no_grad():
                ray_dim = self.num_intervals * self.num_beams * self.nz
                rg = state[:, :ray_dim].reshape(
                    1, self.num_intervals, self.num_beams, self.nz)
                kin = state[:, ray_dim:]
                if self.stochastic:
                    action = self.actor.sample(rg, kin)
                else:
                    action = self.actor.get_mean_action(rg, kin)
            action = np.clip(action.detach().cpu().numpy().reshape(-1), -1.0, 1.0)

            if distance <= self.goal_tolerance:
                if not self.arrived:
                    rospy.loginfo("cnav: goal reached (%.2f m)", distance)
                self.arrived = True
                self._reset_command_filter()
                self.vel_pub.publish(Twist())
                rate.sleep()
                continue

            self.arrived = False
            self.vel_pub.publish(self._command_to_twist(action))

            rate.sleep()

        self._reset_command_filter()
        self.vel_pub.publish(Twist())

    # -------------------------------------------------------------- debug viz
    def _publish_goal_marker(self):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = rospy.Time.now()
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.scale.x = m.scale.y = 0.4
        m.scale.z = 0.05
        m.color.a = 1.0
        m.color.g = 1.0
        m.pose.position.x = self.goal[0]
        m.pose.position.y = self.goal[1]
        m.pose.orientation.w = 1.0
        self.marker_pub.publish(m)

    def _publish_cost_ray_path(self, rays):
        """Draw the theta=0 layer of the cost rays as a polar profile, so the
        perceived free space is visible in rviz.  `env`-frame beams advance
        clockwise in ROS, hence the sign flip."""
        path = Path()
        path.header.frame_id = self.base_frame
        path.header.stamp = rospy.Time.now()
        profile = rays[0, :, 0].detach().cpu().numpy()
        sign = -1.0 if self.mirror_y else 1.0
        for b, d in enumerate(profile):
            ang = sign * math.radians(b * self.h_res_deg)
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(min(d, self.max_range) * math.cos(ang))
            ps.pose.position.y = float(min(d, self.max_range) * math.sin(ang))
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.plan_pub.publish(path)

    def _publish_cost_ray_image(self, cr4):
        """Render normalized first-entry distance as an rgb8 debug image.

        The actor consumes the configured ray-segment channels. This RViz
        image displays channel 0 only: columns are beams rolled so that beam 0
        (the robot's front) is the centre column, rows are theta layers, red
        means a nearby entry and green means clear/far.
        """
        cell = self.cost_ray_image_cell
        proc = cr4[0, 0].detach().cpu().numpy()                 # (B, nz)
        v = np.clip(proc, 0.0, 1.0)                             # (B, nz)
        # Roll the beam axis by half a turn so the robot's FRONT (beam 0) is the
        # centre column and its rear falls on the two edges, matching the HUD
        # panel in SAC/render_utils.draw_cost_rays.  Bearing increases to the
        # robot's right, so the left half of the image is its left side.
        # Display only: the actor consumes the unrolled beam order.
        v = np.roll(v, self.num_beams // 2, axis=0)
        # Transposing to (nz, B, 3) puts the beam on the image x axis and the
        # heading layer on y, the orientation play_cspace draws it in.
        rgb = np.empty((self.nz, self.num_beams, 3), dtype=np.uint8)
        rgb[:, :, 0] = (255.0 * (1.0 - v.T)).astype(np.uint8)
        rgb[:, :, 1] = (255.0 * v.T).astype(np.uint8)
        rgb[:, :, 2] = 40
        # Nearest-neighbour upscale; no PIL or cv_bridge needed for that.
        big = np.repeat(np.repeat(rgb, cell, axis=0), cell, axis=1)
        img = Image()
        img.header.frame_id = self.base_frame
        img.header.stamp = rospy.Time.now()
        img.height, img.width = big.shape[0], big.shape[1]
        img.encoding = "rgb8"
        img.is_bigendian = 0
        img.step = img.width * 3
        img.data = big.tobytes()
        self.cost_ray_image_pub.publish(img)
