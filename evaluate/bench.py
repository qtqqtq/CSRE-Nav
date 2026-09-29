#!/usr/bin/env python3

"""Run one planner through the targets hand-written in the world config.

    ./evaluate/run_bench.py cspace
    ./evaluate/run_bench.py neupan

Started by run_bench.py, which owns the simulation; this file assumes a live
Gazebo and a loaded planner. One planner at a time: the two nodes both
want /cmd_vel and there is one robot.

WHAT IS MEASURED, AND WHY IT IS MEASURED THIS WAY

Ground truth is /gazebo/model_states, never either planner's own odometry -- a
planner that mis-locates itself should not get to grade its own homework.  The
Limo's skid-steer plugin publishes its world pose directly as odom, so spawning
and teleporting keep odom == world and this costs nothing (bench.py asserts it
in pre-flight rather than trusting it).

The trial clock starts at MOTION ONSET, not at goal publication.  cspace loads a
torch checkpoint and neupan builds an initial path; both differ in how long they
take to start moving, and that latency is not navigation quality.

The goal is published AFTER the per-trial teleport, and this order matters.
neupan's goal_callback calls update_initial_path_from_goal(self.robot_state,
goal), anchoring its straight initial path at wherever the robot is when the
message lands -- so publishing first would anchor trial N's path at the end pose
of trial N-1.  The cost is a few milliseconds in which the planner is still
chasing the previous goal after the teleport; that shows up as start_offset,
which is reported per trial rather than hidden.

cspace has a failure mode that is invisible from the outside: its one and only
tf lookup lives in scan_callback, and on failure it logs (throttled) and
returns, so the occupancy grid never updates and the robot simply never moves --
no error, no crash, nothing but a WARN.  That is why this harness gates on
waitForTransform for base_link->laser_link in pre-flight and scores no_motion
separately from timeout.
"""

import argparse
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))

from bench_common import (footprint_clearance, goals_from_config, goals_sha1,
                          load_bench_config, load_world_config, parse_footprint,
                          point_in_polygon, polygon_extent, polygon_ray_exit,
                          sha1_of, trial_plan, world_config_path)

import rospy
import tf
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Log
from sensor_msgs.msg import LaserScan


# Per-planner wiring.  `nodes` is a prefix match because roslaunch resolves the
# final node name and a suffix could be appended; `ready` is retained only as a
# diagnostic record -- the real readiness handshake is observable behaviour
# (see _await_goal_accepted), because /rosout is fire-and-forget and a harness
# that starts after the planner has already logged would wait forever.
PLANNERS = {
    "cspace": {
        "nodes": ("cnav",),
        "goal_topic": "/cnav_goal",
        "claimed_arrival": ("cnav: goal reached",),
        "goal_frame": "odom",
    },
    "neupan": {
        "nodes": ("neupan",),
        "goal_topic": "/move_base_simple/goal",
        "claimed_arrival": ("arrive at the target",),
        "goal_frame": "odom",
    },
}

MODEL_NAME = "limo"
BASE_FRAME = "base_link"
LIDAR_FRAME = "laser_link"
ODOM_FRAME = "odom"


def log(msg):
    rospy.loginfo("[bench] %s" % msg)


def now_s():
    """Sim time.  Only meaningful once /clock is flowing -- before the first
    /clock message rospy.Time.now() is 0, and a deadline computed from it lands
    in the past the moment the clock arrives."""
    return rospy.Time.now().to_sec()


def wall_s():
    """Wall time, for the startup handshake only.

    Pre-flight asks "is the ROS graph up yet", which is not a question about
    simulated dynamics, and sim time cannot answer it: it is 0 until /clock
    arrives (so a deadline is born already expired when the harness attaches to
    a sim that has been running for a while) and it freezes if the sim is
    paused (so the wait never ends).  Trial timing stays on sim seconds, where
    the robot's own dynamics are the clock that matters.
    """
    return time.time()


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.z * q.z + q.y * q.y))


def quat_of(yaw):
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


def _xy(g):
    return "(%.3f, %.3f)" % (g[0], g[1]) if g is not None else "(no pose)"


def median(xs):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def local_speed_from_trace(rows):
    """Mean achieved speed, differenced from the ground-truth trace.

    Deliberately not from /cmd_vel.  A planner can hold a command for a minute
    while the robot never moves -- cspace does exactly that on goals outside the
    distance band it trained on -- and a commanded-speed average calls that
    0.18 m/s.  This is the speed the robot actually had.

    Returns None when the trace is too short or its timestamps do not advance,
    rather than inventing a zero.
    """
    if len(rows) < 2:
        return None
    total = 0.0
    span = 0.0
    for a, b in zip(rows, rows[1:]):
        dt = b["t"] - a["t"]
        if dt <= 0.0:
            continue
        total += math.hypot(b["x"] - a["x"], b["y"] - a["y"])
        span += dt
    if span <= 0.0:
        return None
    return total / span


class Bench(object):
    def __init__(self, args, cfg, wcfg, footprint, body, laser_xy):
        self.args = args
        # Two configs, kept apart on purpose: `cfg` is the harness (timeouts,
        # success radius, footprint) and `wcfg` is the map (arena bounds, start
        # pose, targets).  Mixing them is how a trial timeout ends up needing a
        # world edit.
        self.cfg = cfg
        self.wcfg = wcfg
        # The robot's shape and where its laser sits inside it.  Both are in the
        # robot frame, which is what makes the collision test below need no
        # world pose at all.  `footprint` is what a collision is scored against;
        # `body` is the physical chassis, used only to recognise the laser's own
        # returns -- see scan_verdict().
        self.footprint = footprint
        self.body = body
        self.laser_xy = laser_xy
        self.bounds = tuple(wcfg["arena"]["bounds"])
        self.start_pose = tuple(args.start)
        self.planner = PLANNERS[args.planner]
        self.claimed_arrival = False

        # latest samples, filled by the callbacks below
        self._ms = None
        self._odom = None
        self._scan = None
        self._cmd = None
        self._cmd_stamp = -1.0
        self._ms_stamp = -1.0

        # raw ground-truth trace for the active trial (full model_states rate,
        # used for path length; the JSON keeps the decimated log instead)
        self._trace = []
        self._active = False

        self.goal_pub = rospy.Publisher(
            self.planner["goal_topic"], PoseStamped, queue_size=1)
        self._sms = None

        rospy.Subscriber("/gazebo/model_states", ModelStates, self._on_ms)
        rospy.Subscriber("/odom", Odometry, self._on_odom)
        rospy.Subscriber("/limo/scan", LaserScan, self._on_scan)
        rospy.Subscriber("/cmd_vel", Twist, self._on_cmd)
        rospy.Subscriber("/rosout", Log, self._on_log)

    # ------------------------------------------------------------- callbacks
    def _on_ms(self, msg):
        self._ms = msg
        self._ms_stamp = now_s()
        if self._active and MODEL_NAME in msg.name:
            p = msg.pose[msg.name.index(MODEL_NAME)]
            self._trace.append((now_s(), p.position.x, p.position.y))

    def _on_odom(self, msg):
        self._odom = msg

    def _on_scan(self, msg):
        self._scan = msg

    def _on_cmd(self, msg):
        self._cmd = msg
        self._cmd_stamp = now_s()

    def _on_log(self, msg):
        if not any(msg.name.startswith(n) for n in self.planner["nodes"]):
            return
        if any(s in msg.msg for s in self.planner["claimed_arrival"]):
            self.claimed_arrival = True

    # --------------------------------------------------------------- lookups
    def gt(self):
        """Ground-truth pose of the robot from Gazebo, in world == odom frame."""
        msg = self._ms
        if msg is None or MODEL_NAME not in msg.name:
            return None
        p = msg.pose[msg.name.index(MODEL_NAME)]
        return (p.position.x, p.position.y, yaw_of(p.orientation))

    def moving_models(self):
        """Live Gazebo positions of the world's moving obstacles."""
        if self._ms is None:
            return []
        return [(name, pose.position.x, pose.position.y)
                for name, pose in zip(self._ms.name, self._ms.pose)
                if name.startswith('cross_clone_')]

    def scan_verdict(self):
        """Collision, read directly off the laser.

        The robot's footprint and the beam bearings are both expressed in the
        robot frame -- the laser is bolted to it -- so for each beam the only
        question is whether the return came back shorter than the footprint's
        own radius in that direction.  A shorter one terminated *inside* the
        robot, so what it hit was the robot's body, not an obstacle.  No world
        pose, no odometry and no obstacle list take part, which is the whole
        point: the verdict is a sensor reading, not a geometric reconstruction.

        Returns a dict, or ``None`` before the first scan arrives:

            collision  bool    a beam returned at or inside the footprint
            margin     float   distance from the footprint edge to the nearest
                               return, negative when a beam ended inside it
            bearing    float   that beam's bearing, degrees, robot frame
            range      float   its raw range
            self_d     float   the footprint radius it was compared against
            n_used     int     beams that took part
            n_skipped  int     beams dropped as the robot seeing its own body

        A beam whose return lands inside the physical chassis (`robot.body`) is
        dropped first: that is the laser looking at the robot, not at the world.
        It is not a nicety.  The scored footprint is 0.140 m wide and the real
        body 0.220 m, so the robot sticks out past its own silhouette and its
        chassis sits a centimetre or so *outside* it on nearly every bearing --
        left in, that artefact is the smallest margin on almost every scan and
        the first slightly-closer self-return ends the trial.  `body` is the
        union of the URDF's collision primitives, so the real chassis is inside
        it and every genuine obstacle is outside it.

        The scan is published at 8 Hz and this runs at the trial log rate, so the
        sample may be up to ~0.12 s old, i.e. ~6 cm of travel at 0.5 m/s.
        """
        msg = self._scan
        if msg is None:
            return None

        checks = self.cfg["trial_checks"]
        slack = float(checks.get("scan_margin", 0.0) or 0.0)
        ox, oy = self.laser_xy

        worst = None
        n_used = n_skipped = 0
        for i, r in enumerate(msg.ranges):
            if not (r == r) or r < msg.range_min or r > msg.range_max:
                continue                      # NaN, or off the sensor's scale
            theta = msg.angle_min + i * msg.angle_increment
            body = polygon_ray_exit(ox, oy, theta, self.body)
            if body is not None and r <= body:
                n_skipped += 1
                continue                      # the robot's own chassis
            d = polygon_ray_exit(ox, oy, theta, self.footprint)
            if d is None:
                continue                      # laser outside the footprint
            n_used += 1
            if worst is None or r - d < worst[0]:
                worst = (r - d, math.degrees(theta), r, d)

        if worst is None:
            return None
        return {
            "collision": worst[0] <= slack,
            "margin": worst[0],
            "bearing": worst[1],
            "range": worst[2],
            "self_d": worst[3],
            "n_used": n_used,
            "n_skipped": n_skipped,
        }

    def start_clearance(self):
        """Signed clearance from the footprint, at the start pose, to the moving
        obstacles.  A world-frame question the laser cannot answer: it is still
        mounted on a robot that has not been teleported to that pose yet."""
        return footprint_clearance(self.start_pose[0], self.start_pose[1],
                                   self.start_pose[2], self.footprint,
                                   self.moving_models())

    def scan_min(self):
        """Raw minimum return, unfiltered -- a diagnostic, not the verdict.

        Kept in the trial log because it is what shows the self-returns: on a
        clean run it sits at the sensor floor for a large fraction of samples
        while `scan_margin` stays comfortably positive.
        """
        if self._scan is None:
            return float("nan")
        r = [x for x in self._scan.ranges if x == x and 0.05 < x < 50.0]
        return min(r) if r else float("nan")

    def odom(self):
        if self._odom is None:
            return None
        p = self._odom.pose.pose
        return (p.position.x, p.position.y, yaw_of(p.orientation))

    def cmd(self):
        if self._cmd is None:
            return (0.0, 0.0)
        return (self._cmd.linear.x, self._cmd.angular.z)

    def is_moving(self):
        """Is the planner commanding motion?  moving_speed is a near-zero
        threshold applied to both linear (m/s) and angular (rad/s); both planners
        publish an exact zero Twist on arrival, so anything above it is a command
        to move, and units do not need to agree to mean 'essentially zero'."""
        v, w = self.cmd()
        eps = self.cfg["trial"]["moving_speed"]
        return abs(v) > eps or abs(w) > eps

    # ------------------------------------------------------------ pre-flight
    def preflight(self):
        # Deliberately no self.cfg["trial"] timeouts here: every one of them is
        # in sim seconds, and pre-flight runs before sim time can be trusted.
        if not rospy.get_param("/use_sim_time", False):
            rospy.logfatal("[bench] /use_sim_time is not set.  Every timeout in "
                           "bench.yaml is in SIM seconds; running on wall time "
                           "would silently misjudge all of them. Start the sim "
                           "with run_bench.py first.")
            return False

        # Wall-clock deadline, not now_s(): see wall_s().  Computing this from
        # sim time is what made the harness fail outright when attached to a sim
        # that had been up for a while -- the deadline was born already expired.
        deadline = wall_s() + 120.0

        # /clock first, because everything downstream counts in sim seconds and
        # because a clock stuck at 0 makes trial timeouts that never expire.
        while now_s() == 0.0 and wall_s() < deadline and not rospy.is_shutdown():
            rospy.sleep(0.1)
        if now_s() == 0.0:
            rospy.logfatal("[bench] sim time is still 0 after %g s of wall time: "
                           "either /clock is not being published or Gazebo is "
                           "paused.  Every timeout here is in sim seconds, so "
                           "there is no safe way to continue.", 120.0)
            return False
        log("  /clock OK (sim t=%.1f s)" % now_s())

        log("pre-flight: waiting for topics")
        for name, ok in (("/gazebo/model_states", lambda: self._ms is not None),
                         ("/odom", lambda: self._odom is not None),
                         ("/limo/scan", lambda: self._scan is not None)):
            while not ok() and wall_s() < deadline and not rospy.is_shutdown():
                rospy.sleep(0.1)
            if not ok():
                rospy.logfatal("[bench] timed out waiting for %s" % name)
                return False
            log("  %s OK" % name)

        while len(self.moving_models()) < 20 and wall_s() < deadline:
            rospy.sleep(0.1)
        movers = self.moving_models()
        if len(movers) != 20:
            rospy.logfatal("[bench] expected 20 moving obstacles, found %d; "
                           "check limo_complex.world and Gazebo model paths"
                           % len(movers))
            return False
        before = {name: (x, y) for name, x, y in movers}
        rospy.sleep(2.0)
        motion = max((math.hypot(x - before[name][0], y - before[name][1])
                      for name, x, y in self.moving_models() if name in before),
                     default=0.0)
        if motion < 0.01:
            rospy.logfatal("[bench] 20 obstacles loaded but none moved; "
                           "check rvo_node and set_goals_client")
            return False
        log("  20 moving obstacles OK (max 2 s displacement %.3f m)" % motion)

        # The planner node must be on the master before anything else matters.
        # rospy.get_master().getSystemState() wraps the XML-RPC call, so it comes
        # back as (code, statusMessage, state) and the state is the third
        # element; state itself is [publishers, subscribers, services], each a
        # list of [topic, [node names]] pairs.
        def master_nodes():
            try:
                resp = rospy.get_master().getSystemState()
                state = resp[2] if isinstance(resp[0], int) else resp
                pubs, subs = state[0], state[1]
                out = set()
                for _, owners in list(pubs) + list(subs):
                    out.update(owners)
                return out
            except Exception as exc:
                rospy.logwarn_throttle(5.0, "[bench] master query failed: %s" % exc)
                return set()

        hits = []
        while wall_s() < deadline and not rospy.is_shutdown():
            hits = sorted(n for n in master_nodes()
                          if any(n.lstrip("/").startswith(p)
                                 for p in self.planner["nodes"]))
            if hits:
                break
            rospy.sleep(0.2)
        if not hits:
            rospy.logfatal("[bench] no node matching %s on the master"
                           % (self.planner["nodes"],))
            return False
        log("  planner node OK: %s" % hits)

        pub_deadline = wall_s() + 30.0
        while self.goal_pub.get_num_connections() == 0 and wall_s() < pub_deadline:
            rospy.sleep(0.1)
        if self.goal_pub.get_num_connections() == 0:
            rospy.logfatal("[bench] nothing is subscribed to %s -- the planner "
                           "never created its goal subscriber"
                           % self.planner["goal_topic"])
            return False
        log("  %s has %d subscriber(s) OK"
            % (self.planner["goal_topic"], self.goal_pub.get_num_connections()))

        # The two tf lookups that decide whether either planner works at all.
        # base_link -> laser_link is the one that catches cspace's silent
        # no-op: if it is unavailable cspace still runs and still logs nothing
        # but a throttled INFO while never moving.
        listener = tf.TransformListener()
        for tgt, src in ((ODOM_FRAME, BASE_FRAME), (BASE_FRAME, LIDAR_FRAME)):
            try:
                listener.waitForTransform(tgt, src, rospy.Time(0),
                                          rospy.Duration(20.0))
                listener.lookupTransform(tgt, src, rospy.Time(0))
                log("  tf %s -> %s OK" % (tgt, src))
            except Exception as exc:
                rospy.logfatal("[bench] tf %s -> %s unavailable (%s).  cspace "
                               "returns silently on this and never moves; "
                               "neupan publishes no velocity at all."
                               % (tgt, src, type(exc).__name__))
                return False

        # odom == world is load-bearing: goals are published in odom and the
        # Gazebo ground truth is in world coordinates. If the plugin ever started
        # subtracting the spawn pose these two would drift apart, so published
        # goals would no longer line up with the reported ground-truth pose.
        g, o = self.gt(), self.odom()
        if g is None:
            rospy.logfatal("[bench] no model named %r in /gazebo/model_states "
                           "(saw %s)" % (MODEL_NAME, list(self._ms.name)))
            return False
        drift = math.hypot(g[0] - o[0], g[1] - o[1])
        log("  ground truth (%.4f, %.4f)  odom (%.4f, %.4f)  drift %.4f m"
            % (g[0], g[1], o[0], o[1], drift))
        if drift > 0.05:
            rospy.logfatal(
                "[bench] odom and Gazebo world disagree by %.3f m.  The bench "
                "assumes the skid-steer plugin publishes its world pose as odom; "
                "if that changed, goals published in odom land somewhere else "
                "than the world-frame goal positions expect." % drift)
            return False

        # Start pose sanity: must be clear of the moving obstacles.  This is the
        # one place the laser cannot answer -- it is mounted on a robot that is
        # not at the start pose yet -- so it stays a world-frame calculation,
        # from the obstacle poses and the configured footprint.
        clr = self.start_clearance()
        log("  start clearance %.3f m (%s requires %.2f)"
            % (clr, os.path.basename(self.args.world_config),
               self.wcfg["arena"]["start_clearance"]))
        if clr < self.wcfg["arena"]["start_clearance"]:
            rospy.logwarn("[bench] moving obstacle is near the start pose; "
                          "continuing because it may move away")

        nearmiss = self.scan_min()
        if not (nearmiss == nearmiss) or nearmiss > 3.0:
            rospy.logwarn("[bench] nearest lidar return is %.2f m at the start "
                          "pose; the world may not have loaded" % nearmiss)
        return True

    # --------------------------------------------------------------- actions
    def teleport(self, pose):
        """Return the robot to a pose, then confirm Gazebo actually did it."""
        if self._sms is None:
            rospy.wait_for_service("/gazebo/set_model_state", timeout=30.0)
            self._sms = rospy.ServiceProxy("/gazebo/set_model_state",
                                           SetModelState)
        z = self.cfg["trial"].get("spawn_z", 0.01)
        req = ModelState()
        req.model_name = MODEL_NAME
        req.reference_frame = "world"
        req.pose.position.x = pose[0]
        req.pose.position.y = pose[1]
        req.pose.position.z = z
        q = quat_of(pose[2])
        req.pose.orientation.x, req.pose.orientation.y = q[0], q[1]
        req.pose.orientation.z, req.pose.orientation.w = q[2], q[3]
        req.twist = Twist()
        before = self._ms_stamp
        pre = self.gt()
        # The response is logged rather than discarded.  A SetModelState that
        # reports success while the pose does not change means the model has gone
        # inert in the physics engine - a different failure from a rejected
        # request, and one that must not be allowed to look like one.
        try:
            resp = self._sms(req)
            srv_ok, srv_msg = bool(resp.success), resp.status_message
        except rospy.ServiceException as exc:
            srv_ok, srv_msg = False, "ServiceException: %s" % exc

        tol = self.cfg["trial"]["teleport_tolerance"]
        deadline = now_s() + 10.0
        ok = False
        while now_s() < deadline and not rospy.is_shutdown():
            g = self.gt()
            if g is not None and self._ms_stamp > before:
                if (math.hypot(g[0] - pose[0], g[1] - pose[1]) <= tol):
                    ok = True
                    break
            rospy.sleep(0.02)
        rospy.loginfo("[bench] teleport -> (%.2f, %.2f, yaw %.2f) z=%.3f  "
                      "service=%s %r  gt %s -> %s  %s"
                      % (pose[0], pose[1], pose[2], z, srv_ok, srv_msg,
                         _xy(pre), _xy(self.gt()),
                         "settled" if ok else "NOT SETTLED"))
        if not ok:
            rospy.logwarn("[bench] teleport to (%.2f, %.2f) did not settle "
                          "within %g m" % (pose[0], pose[1], tol))
        return ok

    def make_goal(self, x, y, yaw):
        m = PoseStamped()
        m.header.frame_id = self.planner["goal_frame"]
        m.header.stamp = rospy.Time.now()
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.orientation.z = math.sin(yaw * 0.5)
        m.pose.orientation.w = math.cos(yaw * 0.5)
        return m

    def publish_goal(self, msg):
        """Send the goal a few times.  A single publish is a race: the goal is
        not latched, and a planner that is mid-cycle can miss it."""
        for _ in range(3):
            msg.header.stamp = rospy.Time.now()
            self.goal_pub.publish(msg)
            rospy.sleep(0.1)

    def await_goal_accepted(self, msg):
        """Block until the planner publishes its first /cmd_vel after the goal.

        This is the readiness handshake, and it is behavioural rather than a
        fixed sleep.  Before a planner accepts its first goal it cannot move the
        robot at all -- cspace publishes nothing without a goal, and neupan
        `continue`s before its publisher while initial_path is None -- so a
        planner that is still loading a checkpoint is indistinguishable from one
        that is idle, except that this will time out.  Re-publishing keeps a goal
        sent before the subscriber existed from being lost.
        """
        t0 = now_s()
        last_republish = t0
        while not rospy.is_shutdown():
            if self._cmd_stamp > t0:
                return True
            t = now_s()
            if t - t0 > self.cfg["trial"]["planner_timeout"]:
                return False
            if t - last_republish > 2.0:
                msg.header.stamp = rospy.Time.now()
                self.goal_pub.publish(msg)
                last_republish = t
            rospy.sleep(0.05)
        return False

    # ---------------------------------------------------------------- trials
    def run_trial(self, goal):
        t = self.cfg["trial"]
        gx, gy = goal["x"], goal["y"]
        self.claimed_arrival = False
        self._trace = []
        self._active = False

        # A moving obstacle may pass through the fixed start between trials.
        # Do not teleport the robot into it and credit the planner with a loss.
        wait_until = now_s() + 30.0
        minimum = self.wcfg["arena"]["start_clearance"]
        while self.start_clearance() < minimum and now_s() < wait_until:
            rospy.sleep(0.1)
        if self.start_clearance() < minimum:
            raise RuntimeError("start pose stayed blocked by moving obstacles "
                               "for 30 sim seconds")
        self.teleport(self.start_pose)

        msg = self.make_goal(gx, gy, goal.get("yaw", 0.0))
        self.publish_goal(msg)
        t_pub = now_s()

        if not self.await_goal_accepted(msg):
            return {
                "i": goal["i"], "goal_i": goal["goal_i"], "goal": goal,
                "verdict": "no_planner",
                "reason": "planner never published /cmd_vel within %g s of the "
                          "goal -- it is not tracking goals at all"
                          % t["planner_timeout"],
                "log": [],
            }

        t_acc = now_s()
        t0 = None            # motion onset
        settle_since = None
        best = None
        last_improve = None
        stuck_at = None
        min_clr = float("inf")
        n_near_miss = 0
        rows = []
        path_len = 0.0
        prev_xy = None
        speeds = []
        self._active = True
        start_xy = self.gt()[:2]
        first_pose = [None]

        rate = rospy.Rate(t["log_rate"])
        outcome = None
        reason = ""
        while not rospy.is_shutdown():
            now = now_s()
            g = self.gt()
            if g is None:
                # pre-flight proved model_states carries the robot, so this can
                # only be a transient gap; give up rather than spin forever.
                if now - t_acc > t["start_timeout"] + t["trial_timeout"]:
                    outcome = "aborted"
                    reason = "lost the ground-truth pose of %r" % MODEL_NAME
                    break
                rate.sleep()
                continue
            x, y, yaw = g
            if first_pose[0] is None:
                first_pose[0] = (x, y, yaw)

            if prev_xy is not None:
                path_len += math.hypot(x - prev_xy[0], y - prev_xy[1])
            prev_xy = (x, y)

            dist = math.hypot(gx - x, gy - y)
            sv = self.scan_verdict()
            clr = sv["margin"] if sv else float("nan")
            if clr == clr and clr < min_clr:
                min_clr = clr
            if sv and clr < self.cfg["trial_checks"]["near_miss_distance"]:
                n_near_miss += 1
            smin = self.scan_min()
            v, w = self.cmd()
            speeds.append(abs(v))

            if t0 is None:
                if sv and sv["collision"]:
                    outcome = "collision"
                    reason = ("laser: a beam returned %.3f m inside the robot's "
                              "footprint at %.0f deg before the trial started"
                              % (-clr, sv["bearing"]))
                    break
                if math.hypot(x - start_xy[0], y - start_xy[1]) > t["motion_epsilon"]:
                    t0 = now
                    last_improve = now
                    best = dist
                elif now - t_acc > t["start_timeout"]:
                    outcome = "no_motion"
                    reason = ("no ground-truth motion within %g s of the planner "
                              "accepting the goal.  For cspace this is the "
                              "signature of its silent tf failure: it logs a "
                              "throttled INFO and returns, so the occ grid never "
                              "updates and it never commands anything."
                              % t["start_timeout"])
                    break
            else:
                elapsed = now - t0
                rows.append({
                    "t": round(elapsed, 3), "x": round(x, 4), "y": round(y, 4),
                    "yaw": round(yaw, 4), "dist": round(dist, 4),
                    "scan_margin": round(clr, 4) if clr == clr else None,
                    "scan_min": round(smin, 4) if smin == smin else None,
                    # Which beam produced the margin.  Kept because the smallest
                    # margin is what decides the verdict, and "how close did it
                    # come, and where" is the first question asked of a collision:
                    # a worst beam that never leaves one bearing, at a range that
                    # barely changes, is the signature of the robot's own body
                    # rather than of traffic.
                    "scan_bearing": round(sv["bearing"], 1) if sv else None,
                    "scan_range": round(sv["range"], 4) if sv else None,
                    "v": round(v, 4), "w": round(w, 4),
                })

                # 1. collision -- read off the laser, against the robot's footprint
                if sv and sv["collision"]:
                    outcome = "collision"
                    reason = ("laser: beam at %.0f deg returned %.3f m, %.3f m "
                              "inside the robot's footprint"
                              % (sv["bearing"], sv["range"], -clr))
                    break

                # 2. success -- inside the radius AND actually stopped there
                if dist <= t["success_radius"] and not self.is_moving():
                    if settle_since is None:
                        settle_since = now
                    elif now - settle_since >= t["success_settle"]:
                        outcome = "success"
                        reason = ("held within %.2f m for %.1f s"
                                  % (t["success_radius"], t["success_settle"]))
                        break
                else:
                    settle_since = None

                # 3. progress bookkeeping (see the stuck note below)
                if dist < best - t["progress_epsilon"]:
                    best = dist
                    last_improve = now
                elif (stuck_at is None and self.is_moving()
                      and now - last_improve > t["stuck_window"]):
                    # Not a terminal verdict by itself.  A legitimate detour
                    # around the divider can take longer than stuck_window
                    # without improving the best distance, and ending the trial
                    # there would turn a slow success into a failure.  Record
                    # when it first happened; only a trial that ends with no
                    # success reports stuck, and the report carries the time.
                    stuck_at = now - t0

                if elapsed > t["trial_timeout"]:
                    outcome = "stuck" if stuck_at is not None else "timeout"
                    reason = ("still not at the goal after %g sim s"
                              % t["trial_timeout"])
                    if stuck_at is not None:
                        reason += ("; best distance stopped improving at %.1f s, "
                                   "so this is a stall rather than a slow approach"
                                   % stuck_at)
                    break

            rate.sleep()

        self._active = False
        duration = (now_s() - t0) if t0 is not None else 0.0
        final = self.gt()
        final_err = math.hypot(gx - final[0], gy - final[1]) if final else float("nan")
        onset_pose = first_pose[0]
        start_offset = (math.hypot(onset_pose[0] - start_xy[0],
                                   onset_pose[1] - start_xy[1])
                        if onset_pose else float("nan"))

        # outcome is set at the break site, so the verdict is never recovered by
        # pattern-matching the prose of `reason`.  Reaching here with it unset
        # means the loop only ever exited via shutdown.
        verdict = outcome if outcome is not None else "aborted"
        achieved = local_speed_from_trace(rows)

        return {
            "i": goal["i"],
            # Which entry of the world config's `goals:` this trial drew.  The
            # same target is drawn by several trials, so this -- not `i` -- is
            # the key to group by.
            "goal_i": goal["goal_i"],
            "goal": goal,
            "verdict": verdict,
            "reason": reason,
            # distance from the nominal start pose to where the robot actually
            # was when the clock started.  Nonzero because the planner is still
            # chasing the previous goal for the few ms between the teleport and
            # the goal landing; reported so it cannot quietly bias a run.
            "start_offset": round(start_offset, 4) if start_offset == start_offset else None,
            "motion_onset_latency": round(t0 - t_pub, 3) if t0 else None,
            "accept_latency": round(t_acc - t_pub, 3),
            "duration": round(duration, 3),
            "final_error": round(final_err, 4) if final_err == final_err else None,
            "path_length": round(self.path_length(t0), 4),
            "min_scan_margin": round(min_clr, 4) if min_clr < float("inf") else None,
            "n_near_miss": n_near_miss,
            "stuck_detected_at": round(stuck_at, 3) if stuck_at is not None else None,
            # Two different quantities that used to share one name.  `speeds`
            # averages the COMMANDED /cmd_vel, which stays healthy while the
            # robot does not move at all -- a planner stalled with its wheels
            # commanded reported a brisk 0.18 m/s.  mean_speed is now the
            # achieved speed, differenced from ground truth; the commanded
            # value is kept beside it so the gap between them stays visible.
            "mean_speed": round(achieved, 4) if achieved is not None else None,
            "mean_cmd_speed": round(sum(speeds) / len(speeds), 4) if speeds else None,
            "planner_claimed_arrival": self.claimed_arrival,
            "n_log_samples": len(rows),
            "log": rows,
        }

    def path_length(self, t0):
        """Distance travelled along the full-rate ground-truth trace.

        Computed here rather than from the decimated log: the log exists to be
        read, this exists to be accurate, and a trial that turns tightly would
        otherwise lose real distance to chord shortening.
        """
        if t0 is None or len(self._trace) < 2:
            return 0.0
        pts = [(x, y) for (ts, x, y) in self._trace if ts >= t0]
        return sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
                   for i in range(1, len(pts)))

    # ------------------------------------------------------------------ main
    def run(self):
        # `args.goals` is the trial plan, not the target list: one entry per
        # episode run, each naming the target it drew (`goal_i`).
        plan = self.args.goals
        log("planner=%s  trials=%d  world=%s"
            % (self.args.planner, len(plan), os.path.basename(self.args.world)))
        results = []
        for goal in plan:
            log("trial %d/%d  target %d -> (%.2f, %.2f) dist %.2f yaw %.2f"
                % (goal["i"] + 1, len(plan), goal["goal_i"], goal["x"],
                   goal["y"], goal["distance"], goal["yaw"]))
            r = self.run_trial(goal)
            log("   %-10s %s" % (r["verdict"].upper(), r.get("reason", "")))
            results.append(r)
            if rospy.is_shutdown():
                break

        counts = {}
        for r in results:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        succ = counts.get("success", 0)
        durs = [r["duration"] for r in results if r["verdict"] == "success"]
        errs = [r["final_error"] for r in results if r["final_error"] is not None]
        self.args.results = results
        self.args.summary = {
            "n_trials": len(results),
            "counts": counts,
            "success_rate": round(float(succ) / len(results), 4) if results else None,
            "time_to_goal_median": round(median(durs), 3) if durs else None,
            "time_to_goal_iqr": ([round(quantile(durs, 0.25), 3),
                                  round(quantile(durs, 0.75), 3)] if len(durs) > 1 else None),
            "final_error_median": round(median(errs), 4) if errs else None,
        }
        log("summary: %s" % json.dumps(self.args.summary, sort_keys=True))
        return results


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planner", required=True, choices=sorted(PLANNERS))
    ap.add_argument("--world", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None,
                    help="harness parameters; default: ../config/bench.yaml "
                         "relative to this script")
    ap.add_argument("--world-config", default=None,
                    help="per-world arena and targets; default: the world file's "
                         "sibling .yaml (worlds/limo_complex.world -> "
                         "worlds/limo_complex.yaml)")
    ap.add_argument("--start", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "YAW"))
    ap.add_argument("--timeout", type=float, default=None,
                    help="override trial.trial_timeout (sim seconds)")
    ap.add_argument("--only", type=int, nargs="*", default=None,
                    help="run only these trial numbers, 0-based, as they appear "
                         "in the plan (debugging)")
    ap.add_argument("--only-goal", type=int, nargs="*", default=None,
                    help="run only the trials that drew these targets, by their "
                         "index in `goals:` (debugging a single target)")
    # rospy.myargv, not a bare argv: when this node is started by roslaunch,
    # roslaunch appends its own reserved arguments (__name:=, __log:=) to the
    # command line, and a strict argparse rejects them -- "unrecognized
    # arguments: __name:=bench_harness".  myargv strips those and any name:=value
    # remapping, which is the ROS 1 idiom for exactly this.  Without it the
    # harness only ever works when invoked by hand, i.e. every path except the
    # also permits direct ROS invocation with remapping arguments.
    args = ap.parse_args(rospy.myargv(argv=argv)[1:])

    pkg = _HERE
    if args.config is None:
        args.config = os.path.join(pkg, "config", "bench.yaml")
    if args.world_config is None:
        args.world_config = world_config_path(args.world)
    cfg = load_bench_config(args.config)
    try:
        wcfg = load_world_config(args.world_config)
    except ValueError as exc:
        sys.exit("[bench] %s" % exc)
    if args.start is None:
        args.start = wcfg["arena"]["start"]
    if args.timeout is not None:
        cfg["trial"]["trial_timeout"] = args.timeout

    try:
        goals = goals_from_config(wcfg,
                                  bounds=tuple(wcfg["arena"]["bounds"]),
                                  start=args.start)
        plan = trial_plan(goals, trials=wcfg.get("trials"),
                          sample=wcfg.get("goal_sample", "cycle"),
                          seed=wcfg.get("trial_seed"))
    except ValueError as exc:
        # Before rospy.init_node, let alone Gazebo: a mistyped target is an
        # authoring mistake, and finding it should not cost a simulation run.
        sys.exit("[bench] %s: %s" % (args.world_config, exc))
    declared = plan
    if args.only is not None:
        plan = [t for t in declared if t["i"] in args.only]
        if not plan:
            sys.exit("[bench] --only %s matched none of the %d trials in %s"
                     % (args.only, len(declared), args.world_config))
    if args.only_goal is not None:
        plan = [t for t in plan if t["goal_i"] in args.only_goal]
        if not plan:
            sys.exit("[bench] --only-goal %s matched none of the %d targets in "
                     "%s" % (args.only_goal, len(goals), args.world_config))
    args.goals = plan
    # The targets, not the schedule: report.py must refuse to compare two runs
    # scored against different points, but must not object to the same points
    # drawn in a different order -- that is what re-running looks like.
    actual_goal_sha1 = goals_sha1(goals)

    rospy.init_node("bench_harness", anonymous=True, disable_signals=True)
    log("starting: planner=%s world_config=%s" % (args.planner, args.world_config))
    log("  %d target(s) -> %d trial(s), sample=%s"
        % (len(goals), len(plan), wcfg.get("goal_sample", "cycle")))
    for g in plan:
        # A target inside the success radius is an instant success and a wasted
        # trial; a legal choice, but almost always a typo in the decimals.
        if g["distance"] < cfg["trial"]["success_radius"]:
            rospy.logwarn("[bench] goals[%d] is %.3f m from the start, within "
                          "the %.2f m success radius -- the trial will succeed "
                          "without the robot moving"
                          % (g["i"], g["distance"], cfg["trial"]["success_radius"]))

    # The shape that scores collisions is the polygon in bench.yaml, which is
    # deliberately NOT derived from the URDF.  The URDF describes the physical
    # chassis (0.29 x 0.22 m including its wheels); the benchmark scores both
    # planners against the 1.20 x 0.14 m envelope cspace was trained to perceive,
    # so the two legitimately disagree and deriving one from the other would
    # silently change what a "collision" means.  See config/bench.yaml.
    try:
        footprint = parse_footprint(cfg["robot"]["footprint"], "robot.footprint")
    except (KeyError, TypeError, ValueError) as exc:
        rospy.logfatal("[bench] %s: %s" % (args.config, exc))
        return 2
    laser_xy = tuple(cfg["robot"].get("laser_xyz", (0.0, 0.0)))[:2]
    flen, fwid = polygon_extent(footprint)
    if not point_in_polygon(laser_xy[0], laser_xy[1], footprint):
        # The collision test compares each return against the footprint's radius
        # from the laser, so a laser outside the polygon has no radius and every
        # beam would be skipped -- a silent "no collision ever" benchmark.
        rospy.logfatal("[bench] robot.laser_xyz %s is outside robot.footprint; "
                       "the laser collision test cannot work.  Check %s."
                       % (laser_xy, args.config))
        return 2
    # The physical chassis, for recognising the laser's own returns.  Required,
    # not defaulted: without it the robot's own body reads as an obstacle and the
    # trial ends on the first scan -- see scan_verdict().
    try:
        body = parse_footprint(cfg["robot"]["body"], "robot.body")
    except (KeyError, TypeError, ValueError) as exc:
        rospy.logfatal("[bench] %s: %s" % (args.config, exc))
        return 2
    blen, bwid = polygon_extent(body)
    if not point_in_polygon(laser_xy[0], laser_xy[1], body):
        rospy.logfatal("[bench] robot.laser_xyz %s is outside robot.body; a laser "
                       "mounted off its own chassis cannot mask its self-returns."
                       % (laser_xy,))
        return 2
    log("footprint: %d point(s), %.3f x %.3f m, laser at (%.3f, %.3f)"
        % (len(footprint), flen, fwid, laser_xy[0], laser_xy[1]))
    log("body mask: %d point(s), %.3f x %.3f m -- returns inside this are the "
        "robot seeing itself" % (len(body), blen, bwid))

    bench = Bench(args, cfg, wcfg, footprint, body, laser_xy)
    if not bench.preflight():
        rospy.logfatal("[bench] pre-flight failed; not running any trial")
        return 2

    try:
        results = bench.run()
    except rospy.ROSInterruptException:
        log("interrupted")
        return 1

    # The hash of the world file this run actually had in front of it.  Not read
    # from anywhere else: metadata about a world is not the world, and two runs
    # have to be judged comparable on the file they were scored against.
    actual_world_sha1 = sha1_of(args.world)
    out = {
        "meta": {
            "planner": args.planner,
            "world": os.path.abspath(args.world),
            "world_sha1": actual_world_sha1,
            # The map's own config (arena + targets), and a hash of the points
            # themselves.  report.py refuses to compare two runs with different
            # goals_sha1: the file may be edited between runs, and a comparison
            # across two different target lists would be an average of nothing.
            "world_config": os.path.abspath(args.world_config),
            "goals_sha1": actual_goal_sha1,
            "start": list(args.start),
            "robot": {"footprint": [list(p) for p in footprint],
                      "extent": [flen, fwid],
                      "body": [list(p) for p in body],
                      "body_extent": [blen, bwid],
                      "laser_xyz": list(laser_xy)},
            "bench_config": os.path.abspath(args.config),
            "trial": cfg["trial"],
            # The schedule, kept so a reader can see which target each trial
            # drew without re-deriving it from the seed.
            "n_trials": len(results),
            "n_targets": len(goals),
            "goal_sample": wcfg.get("goal_sample", "cycle"),
            "trial_seed": wcfg.get("trial_seed"),
            "plan": [{"i": t["i"], "goal_i": t["goal_i"],
                      "x": t["x"], "y": t["y"]} for t in args.goals],
        },
        "summary": args.summary,
        "trials": results,
    }
    d = os.path.dirname(os.path.abspath(args.out))
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    log("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    # Full sys.argv, not sys.argv[1:]: rospy.myargv() keeps argv[0] and filters
    # everything after it, so handing it an already-sliced list makes it treat
    # the first real flag as the program name and drop it.
    sys.exit(main(sys.argv))
