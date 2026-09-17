#!/usr/bin/env python

"""
sac_cspace_core -- ROS 1 wrapper for the CNN-SAC c-space (NavRL) navigation policy.

The policy is the cost-ray CNN actor trained on `Sparrow_V2.SparrowCspaceEnv`.
Its observation is

    state = flatten( preprocess(cost_rays) ) + [distance, heading]
              |__ num_beams x nz cost rays __|  |_____ 2 _____|

where the cost rays are built from the robot's line-of-sight lidar hits, exactly
as `SparrowCspaceEnv._build_occ_grid()` does.  This node rebuilds that observation
from live ROS sensor data using the same pure-numpy/torch helpers in
`SAC/cspace_cost_rays.py`, so it needs neither Gazebo nor the C++ c_space_mapper.

The kinematic block is `[distance (m), heading_error (rad)]`.  It previously also
carried the previous action command (v_lin, v_ang); that feedback was removed from
the training env, so checkpoints trained before the change (feat+4 in `layer_1`)
cannot load here and must be retrained -- see `_load_actor`.

THREE things differ between the checkpoint families in `SAC/pytorch_models/`, and
each is a knob here:

  1. `architecture` -- sniffed out of the state dict, never configured: backbone
     4->8->8 stride 2 vs 4->16->16 stride [1,1]/[1,2], and head
     `layer_3 -> action_dim*2` (stochastic, tanh of the mean) vs
     `layer_3 -> action_dim` (deterministic, tanh inside forward).
  2. `geometry` -- nz / a / g_min / cost_ray_max_range / footprint.  Not
     recoverable from a checkpoint, so it comes from the YAML and is *validated*
     against the loaded network's fc width; a mismatch aborts the node instead of
     silently degrading the policy.
  3. `ray_frame` + `preprocess` + velocity scale -- the Sparrow-trained models read
     a robot-centric grid whose 2nd axis runs to the robot's RIGHT
     (`_build_occ_grid` uses `rel_y = dx*sin(t) + dy*cos(t)`, the mirror of the
     standard ROS body frame) with a log-encoded ray magnitude and a 0.5 m/s /
     2 rad/s action scale; the older Gazebo models read ROS-frame rays from the C++
     mapper with a linear `max_range - d` encoding and the actor output published
     to /cmd_vel unscaled.

Config profiles shipped:

    dense_sample_cspace_*    -> config/sac_cspace_sparrow.yaml        (36x30)
    SAC_sparrow_cspace_*     -> config/sac_cspace_legacy_sparrow.yaml (36x10)
    SAC_cspace_*             -> config/sac_cspace_legacy_gazebo.yaml  (36x10, see caveats)

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
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker


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


# Size of the kinematic block appended to the cost rays: [distance, heading].
# Kept in sync with SAC/settings.py KINEMATIC_DIM.
KINEMATIC_DIM = 2


def _kin_dim_of(state_dict, feature_dim, name=None):
    """Sniff the kinematic-block width out of an actor state dict.

    The block is concatenated with the CNN features and goes into `layer_1`, so
    that layer's input width is the checkpoint's only remaining trace of it.
    Checkpoints trained before the previous-action feedback was removed carried
    `[dist, heading, v_lin, v_ang]` and can no longer be loaded.

    Returns the width, or raises ValueError with a readable message.  Shared by
    the node and check_models.py so the two cannot disagree about the layout.
    """
    kin_dim = state_dict["layer_1.weight"].shape[1] - feature_dim
    if kin_dim != KINEMATIC_DIM:
        raise ValueError(
            "%swas trained with a %d-dim kinematic block (layer_1 expects "
            "feature_dim+%d), but this build uses %d ([distance, heading]). "
            "Checkpoints predating the removal of the previous-action feedback "
            "carried [dist, heading, v_lin, v_ang] and must be retrained."
            % ("checkpoint '%s' " % name if name else "", kin_dim, kin_dim,
               KINEMATIC_DIM))
    return kin_dim


class SacCspaceCore(object):
    """Subscribes to a laser scan + odometry + goal, publishes velocity commands."""

    def __init__(self):
        rospy.init_node("sac_cspace_node", anonymous=True)

        # ---- model ------------------------------------------------------
        self.model_name = rospy.get_param("~model_name", "dense_sample_cspace_2")
        self.sac_dir = rospy.get_param("~sac_dir", "/DRL-robot-navigation/SAC")
        self.model_dir = rospy.get_param("~model_dir", None)
        self.device = torch.device(rospy.get_param("~device", "cuda"))
        self.stochastic = bool(rospy.get_param("~stochastic", False))

        # ---- frames / topics --------------------------------------------
        self.map_frame = rospy.get_param("~map_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.lidar_frame = rospy.get_param("~lidar_frame", "front_laser")
        self.scan_topic = rospy.get_param("~scan_topic", "/r1/front_laser/scan")
        self.odom_topic = rospy.get_param("~odom_topic", "/r1/odom")
        self.cmd_topic = rospy.get_param("~cmd_vel_topic", "/r1/cmd_vel")
        self.goal_topic = rospy.get_param("~goal_topic", "/sac_cspace/goal")
        self.plan_topic = rospy.get_param("~plan_topic", "/sac_cspace/cost_ray_path")
        self.marker_topic = rospy.get_param("~marker_topic", "/sac_cspace/goal_marker")

        # 10 Hz matches the training control interval (Sparrow ctrl_interval = 0.1 s)
        self.rate_hz = float(rospy.get_param("~rate", 10.0))
        self.goal_tolerance = float(rospy.get_param("~goal_tolerance", 0.3))
        self.publish_debug = bool(rospy.get_param("~publish_debug", True))

        # ---- geometry (must match the checkpoint's training run) ---------
        self.g_min = float(rospy.get_param("~g_min", 0.1))
        self.nz = int(rospy.get_param("~nz", 30))
        self.a = float(rospy.get_param("~a", 2.0 * math.pi / 30.0))
        self.max_range = float(rospy.get_param("~cost_ray_max_range", 3.0))
        self.num_beams = int(rospy.get_param("~cost_ray_num_beams", 36))
        self.h_res_deg = float(rospy.get_param("~cost_ray_h_res_degrees", 10.0))
        self.footprint = [[float(x), float(y)] for x, y in rospy.get_param(
            "~footprint", [[0.6, 0.07], [0.6, -0.07], [-0.6, -0.07], [-0.6, 0.07]])]

        # `env`: the grid's 2nd axis runs to the robot's RIGHT, as built by
        #        SparrowCspaceEnv._build_occ_grid (both Sparrow families).
        # `ros`: standard ROS body frame (2nd axis = left); what the C++ c-space
        #        mapper fed the Gazebo-trained models.
        self.ray_frame = rospy.get_param("~ray_frame", "env")
        if self.ray_frame not in ("env", "ros"):
            raise ValueError("~ray_frame must be 'env' or 'ros', got %r" % self.ray_frame)
        self.mirror_y = (self.ray_frame == "env")

        # `log`      : p = ln(max_range / clamp(d, 0.1, max_range))   (Sparrow)
        # `subtract` : p = max_range - max(d, 0.1)                    (old Gazebo)
        self.preprocess = rospy.get_param("~preprocess", "log")
        if self.preprocess not in ("log", "subtract"):
            raise ValueError("~preprocess must be 'log' or 'subtract', got %r"
                             % self.preprocess)

        # ---- action scaling ---------------------------------------------
        # Sparrow: v_phys = continuous_scale * a with (50 cm/s, 2 rad/s).  The
        # Gazebo pipeline published the actor output straight to /cmd_vel, so its
        # scale is 1.0 in m/s and rad/s.
        self.allow_reverse = bool(rospy.get_param("~allow_reverse", True))
        self.v_linear_max = float(rospy.get_param("~v_linear_max", 0.5))
        self.v_angular_max = float(rospy.get_param("~v_angular_max", 2.0))

        # Hits at or beyond this range count as "no return", mirroring the sim's
        # `ld_scan_result < ld_d_range`.  0 => use cost_ray_max_range.
        scan_max = float(rospy.get_param("~scan_max_range", 0.0))
        self.scan_max_range = scan_max if scan_max > 0.0 else self.max_range
        self.scan_min_range = float(rospy.get_param("~scan_min_range", 0.0))

        self._load_helpers()
        self._load_actor()
        self._build_geometry()

        # ---- state ------------------------------------------------------
        self.robot_state = None        # np.array([x, y, yaw]) in `map_frame`
        self.goal = None               # np.array([x, y])
        self.occ_grid = None
        self.arrived = False

        self.vel_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.plan_pub = rospy.Publisher(self.plan_topic, Path, queue_size=1)
        self.marker_pub = rospy.Publisher(self.marker_topic, Marker, queue_size=1)

        self.listener = tf.TransformListener()
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber(self.goal_topic, PoseStamped, self.goal_callback, queue_size=1)

        rospy.loginfo("sac_cspace: model=%s backbone=%s head=%s feature_dim=%d device=%s",
                      self.model_name, self.backbone, self.action_head,
                      self.feature_dim, self.device)
        rospy.loginfo("sac_cspace: geometry nz=%d a=%.4f g_min=%.3f range=%.2f beams=%d "
                      "-> ray_grid=%dx%d, occ_grid=%dx%d, state_dim=%d",
                      self.nz, self.a, self.g_min, self.max_range, self.num_beams,
                      self.num_beams, self.nz, self.G, self.G,
                      self.num_beams * self.nz + KINEMATIC_DIM)
        rospy.loginfo("sac_cspace: ray_frame=%s preprocess=%s scale=(%.2f m/s, %.2f rad/s) "
                      "rate=%.1f Hz", self.ray_frame, self.preprocess,
                      self.v_linear_max, self.v_angular_max, self.rate_hz)
        rospy.loginfo("sac_cspace: waiting for a goal on %s", self.goal_topic)

    # ------------------------------------------------------------------ setup
    def _load_helpers(self):
        """Import the pure-numpy cost-ray helpers from the SAC training tree."""
        if self.sac_dir not in sys.path:
            sys.path.insert(0, self.sac_dir)
        try:
            from cspace_cost_rays import (build_cspace, cost_rays_3d,
                                          make_footprint_masks,
                                          preprocess_cost_rays)
        except ImportError as exc:
            raise ImportError(
                "cannot import cspace_cost_rays from sac_dir='%s' (%s). Set the "
                "~sac_dir parameter to the DRL-robot-navigation/SAC directory."
                % (self.sac_dir, exc))
        self._build_cspace = build_cspace
        self._cost_rays_3d = cost_rays_3d
        self._make_masks = make_footprint_masks
        self._preprocess = preprocess_cost_rays

    def _load_actor(self):
        """Load <model_name>_actor.pth, building a network to match its shapes."""
        if self.model_dir is None:
            self.model_dir = os.path.join(self.sac_dir, "pytorch_models")
        path = os.path.join(self.model_dir, "%s_actor.pth" % self.model_name)
        if not os.path.exists(path):
            raise IOError("actor checkpoint not found: %s" % path)

        state = torch.load(path, map_location="cpu")
        self.backbone, self.action_head, self.feature_dim = _arch_of(state)

        backbone_cls = _BackboneNew if self.backbone == "new" else _BackboneOld
        actor_cls = (_ActorStochastic if self.action_head == "stochastic"
                     else _ActorDeterministic)
        actor = actor_cls(
            backbone_cls(in_channels=1, h_beams=self.num_beams, v_layers=self.nz,
                         feature_dim=self.feature_dim),
            self.feature_dim, KINEMATIC_DIM, 2)

        # Same idea for the kinematic block: a checkpoint from before the
        # previous-action feedback was removed expects feature_dim + 4, and would
        # otherwise fail with an opaque shape mismatch on layer_1.
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
        """One-time footprint masks + robot-centric grid geometry.

        Mirrors `SparrowCspaceEnv.__init__`: R = ceil(max_range/g_min) + 2, the grid
        is (2R+1)^2, and the robot always sits at cell (R, R).
        """
        self.masks = torch.as_tensor(
            self._make_masks(self.g_min, self.a, self.nz, self.footprint),
            dtype=torch.float32, device=self.device)
        self.R = int(math.ceil(self.max_range / self.g_min)) + 2
        self.G = 2 * self.R + 1
        self.robot_cell = torch.full((1, 2), self.R, dtype=torch.long,
                                     device=self.device)

    # ------------------------------------------------------------- callbacks
    def odom_callback(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.z * q.z + q.y * q.y))
        self.robot_state = np.array([p.x, p.y, yaw], dtype=np.float64)

    def goal_callback(self, msg):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y],
                             dtype=np.float64)
        self.arrived = False
        rospy.loginfo("sac_cspace: new goal (%.2f, %.2f)", self.goal[0], self.goal[1])
        if self.publish_debug:
            self._publish_goal_marker()

    def scan_callback(self, msg):
        """Rasterise line-of-sight lidar hits into the robot-centric occ grid.

        Mirrors `SparrowCspaceEnv._build_occ_grid`: only a ray's *endpoint* cell is
        marked occupied (there is no ray-traced clearing), and a ray that reaches
        the horizon hit nothing and is dropped.
        """
        if self.robot_state is None:
            return

        ranges = np.asarray(msg.ranges, dtype=np.float64)
        if ranges.size == 0:
            return
        angles = msg.angle_min + np.arange(ranges.size) * msg.angle_increment

        lo = max(msg.range_min, self.scan_min_range)
        hi = min(msg.range_max, self.scan_max_range)
        hit = np.isfinite(ranges) & (ranges > lo) & (ranges < hi)
        if not np.any(hit):
            self.occ_grid = None
            return

        r, th = ranges[hit], angles[hit]

        # lidar -> base_link (this is what carries the robot's heading)
        try:
            (trans, rot) = self.listener.lookupTransform(
                self.base_frame, self.lidar_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException):
            rospy.loginfo_throttle(2.0, "sac_cspace: waiting for tf %s -> %s",
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

        g, R, G = self.g_min, self.R, self.G
        ci = R + np.round(rel_x / g).astype(np.int64)
        # The trained grid's axis 1 runs to the robot's RIGHT, so the ROS left
        # component is negated before quantisation (see `ray_frame`).
        j = -rel_y if self.mirror_y else rel_y
        cj = R + np.round(j / g).astype(np.int64)

        inb = (ci >= 0) & (ci < G) & (cj >= 0) & (cj < G)
        if not np.any(inb):
            self.occ_grid = None
            return

        occ = np.zeros((G, G), dtype=bool)
        occ[ci[inb], cj[inb]] = True
        self.occ_grid = occ

    # ------------------------------------------------------------- inference
    def _build_state(self):
        """Assemble the (1, B*nz + KINEMATIC_DIM) observation plus the raw rays."""
        if self.occ_grid is None:
            # No usable return: the world is free within the horizon, which is
            # exactly what an empty occupancy grid encodes.
            occ = torch.zeros((1, self.G, self.G), dtype=torch.bool, device=self.device)
        else:
            occ = torch.as_tensor(self.occ_grid[None, ...], dtype=torch.bool,
                                  device=self.device)

        cspace = self._build_cspace(occ, self.masks)
        rays = self._cost_rays_3d(
            cspace, self.robot_cell,
            num_beams=self.num_beams, h_res_deg=self.h_res_deg,
            max_range=self.max_range, g_min=self.g_min)          # (1, B, nz) m

        if self.preprocess == "log":
            cr = self._preprocess(rays, max_range=self.max_range)
        else:
            cr = (self.max_range - rays.clamp(min=0.1)).unsqueeze(1)
        cr = cr.reshape(1, -1).float()

        dx = self.goal[0] - self.robot_state[0]
        dy = self.goal[1] - self.robot_state[1]
        distance = math.hypot(dx, dy)
        # Standard ROS bearing error; numerically identical to the env's
        # atan2(robot - goal) + pi/2 - yaw formulation.
        heading = wrap_to_pi(math.atan2(dy, dx) - self.robot_state[2])

        kin = torch.tensor([[distance, heading]],
                           dtype=torch.float32, device=self.device)
        return torch.cat([cr, kin], dim=1), rays, distance

    def _command_to_twist(self, action):
        """Actor output in [-1,1] -> physical Twist (m/s, rad/s)."""
        a0 = float(action[0])
        if not self.allow_reverse:
            a0 = (a0 + 1.0) / 2.0
        twist = Twist()
        twist.linear.x = self.v_linear_max * a0
        twist.angular.z = self.v_angular_max * float(action[1])
        return twist

    def run(self):
        rate = rospy.Rate(self.rate_hz)

        while not rospy.is_shutdown():
            if self.robot_state is None or self.goal is None:
                if self.goal is None:
                    rospy.logwarn_throttle(
                        5.0, "sac_cspace: no goal yet -- publish a PoseStamped on %s",
                        self.goal_topic)
                rate.sleep()
                continue

            state, rays, distance = self._build_state()

            with torch.no_grad():
                rg = state[:, :self.num_beams * self.nz].reshape(
                    1, 1, self.num_beams, self.nz)
                kin = state[:, self.num_beams * self.nz:]
                if self.stochastic:
                    action = self.actor.sample(rg, kin)
                else:
                    action = self.actor.get_mean_action(rg, kin)
            action = np.clip(action.detach().cpu().numpy().reshape(-1), -1.0, 1.0)

            if distance <= self.goal_tolerance:
                if not self.arrived:
                    rospy.loginfo("sac_cspace: goal reached (%.2f m)", distance)
                self.arrived = True
                self.vel_pub.publish(Twist())
                rate.sleep()
                continue

            self.arrived = False
            self.vel_pub.publish(self._command_to_twist(action))

            if self.publish_debug:
                self._publish_cost_ray_path(rays)

            rate.sleep()

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
