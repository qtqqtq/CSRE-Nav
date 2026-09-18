#!/usr/bin/env python3

"""Run one planner through the shared goal set and score every trial.

    roslaunch planner_bench bench.launch planner:=cspace
    roslaunch planner_bench bench.launch planner:=neupan

Started by bench.launch, not by hand.  One planner at a time: the two nodes both
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

# installed, this script sits next to bench_common.py in lib/<pkg>/; run from the
# source tree, the module is one directory over in src/
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, os.pardir, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench_common import (footprint_from_urdf, load_bench_config, min_clearance,
                          parse_world_obstacles, point_aabb_distance, sha1_of)

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
    def __init__(self, args, cfg, obstacles, half_len, half_wid):
        self.args = args
        self.cfg = cfg
        self.obstacles = obstacles
        self.half_len = half_len
        self.half_wid = half_wid
        self.bounds = tuple(cfg["arena"]["bounds"])
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

    def odom(self):
        if self._odom is None:
            return None
        p = self._odom.pose.pose
        return (p.position.x, p.position.y, yaw_of(p.orientation))

    def scan_min(self):
        if self._scan is None:
            return float("nan")
        r = [x for x in self._scan.ranges if x == x and 0.05 < x < 50.0]
        return min(r) if r else float("nan")

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
                           "would silently misjudge all of them.  Is sim.launch up?")
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
        # world-file AABBs are in world coordinates.  If the plugin ever started
        # subtracting the spawn pose these two would drift apart and every
        # collision and success verdict would be silently wrong.
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
                "than the world AABBs expect." % drift)
            return False

        # Start pose sanity: must be inside the arena and clear of obstacles.
        clr = min_clearance(self.start_pose[0], self.start_pose[1],
                            self.start_pose[2], self.half_len, self.half_wid,
                            self.obstacles)
        log("  start clearance %.3f m (bench.yaml requires %.2f)"
            % (clr, self.cfg["arena"]["start_clearance"]))
        if clr < self.cfg["arena"]["start_clearance"]:
            rospy.logfatal("[bench] start pose is too close to an obstacle")
            return False

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

        self.teleport(self.start_pose)

        msg = self.make_goal(gx, gy, goal.get("yaw", 0.0))
        self.publish_goal(msg)
        t_pub = now_s()

        if not self.await_goal_accepted(msg):
            return {
                "i": goal["i"], "goal": goal, "verdict": "no_planner",
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
            clr = min_clearance(x, y, yaw, self.half_len, self.half_wid,
                                self.obstacles)
            if clr < min_clr:
                min_clr = clr
            smin = self.scan_min()
            if smin == smin and smin < self.cfg["trial_checks"]["near_miss_distance"]:
                n_near_miss += 1
            v, w = self.cmd()
            speeds.append(abs(v))

            if t0 is None:
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
                    "clearance": round(clr, 4),
                    "scan_min": round(smin, 4) if smin == smin else None,
                    "v": round(v, 4), "w": round(w, 4),
                })

                # 1. collision -- the robot's true footprint against the true AABBs
                if clr <= 0.0:
                    outcome = "collision"
                    reason = ("footprint penetrated an obstacle AABB by %.3f m "
                              "(ground truth, not the planner's odometry)"
                              % -clr)
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
            "min_clearance": round(min_clr, 4) if min_clr < float("inf") else None,
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
        goals = self.args.goals_doc["goals"]
        log("planner=%s  goals=%d  world=%s"
            % (self.args.planner, len(goals), os.path.basename(self.args.world)))
        results = []
        for goal in goals:
            log("trial %d/%d -> (%.2f, %.2f) dist %.2f los %.2f band %s"
                % (goal["i"] + 1, len(goals), goal["x"], goal["y"],
                   goal["distance"], goal["los_clearance"], goal["band"]))
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
    ap.add_argument("--goals", required=True)
    ap.add_argument("--world", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None,
                    help="default: ../config/bench.yaml relative to this script")
    ap.add_argument("--start", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "YAW"))
    ap.add_argument("--timeout", type=float, default=None,
                    help="override trial.trial_timeout (sim seconds)")
    ap.add_argument("--only", type=int, nargs="*", default=None,
                    help="run only these goal indices (debugging)")
    # rospy.myargv, not a bare argv: when this node is started by roslaunch,
    # roslaunch appends its own reserved arguments (__name:=, __log:=) to the
    # command line, and a strict argparse rejects them -- "unrecognized
    # arguments: __name:=bench_harness".  myargv strips those and any name:=value
    # remapping, which is the ROS 1 idiom for exactly this.  Without it the
    # harness only ever works when invoked by hand, i.e. every path except the
    # documented `roslaunch planner_bench bench.launch`.
    args = ap.parse_args(rospy.myargv(argv=argv)[1:])

    pkg = os.path.normpath(os.path.join(_HERE, os.pardir))
    if args.config is None:
        args.config = os.path.join(pkg, "config", "bench.yaml")
    cfg = load_bench_config(args.config)
    if args.start is None:
        args.start = cfg["arena"]["start"]
    if args.timeout is not None:
        cfg["trial"]["trial_timeout"] = args.timeout

    with open(args.goals) as fh:
        args.goals_doc = json.load(fh)
    if args.only:
        args.goals_doc["goals"] = [g for g in args.goals_doc["goals"]
                                   if g["i"] in args.only]

    rospy.init_node("bench_harness", anonymous=True, disable_signals=True)
    log("starting: planner=%s goals=%s" % (args.planner, args.goals))

    obstacles = parse_world_obstacles(args.world, margin=0.0)
    log("world: %d obstacle AABBs from %s"
        % (len(obstacles), os.path.basename(args.world)))

    # The footprint that scores collisions comes from the live URDF, not from
    # bench.yaml: a collision verdict is a physical claim, and it should be made
    # against the robot that is actually in the simulation.
    half_len, half_wid = cfg["robot"]["half_length"], cfg["robot"]["half_width"]
    try:
        urdf = rospy.get_param("/robot_description")
        hl, hw, ncol = footprint_from_urdf(urdf)
        tol = cfg["robot"]["tolerance"]
        if abs(hl - half_len) > tol or abs(hw - half_wid) > tol:
            rospy.logwarn(
                "[bench] /robot_description footprint is %.3f x %.3f m but "
                "bench.yaml says %.3f x %.3f m; using the URDF.  Collision "
                "verdicts would otherwise be made against a robot that is not "
                "in the simulation." % (2 * hl, 2 * hw, 2 * half_len, 2 * half_wid))
        else:
            log("footprint from URDF: %.4f x %.4f m (%d collision primitives) "
                "- agrees with bench.yaml" % (2 * hl, 2 * hw, ncol))
        half_len, half_wid = hl, hw
    except Exception as exc:
        rospy.logwarn("[bench] could not derive the footprint from "
                      "/robot_description (%s: %s); falling back to bench.yaml "
                      "(%.3f x %.3f m)"
                      % (type(exc).__name__, exc, 2 * half_len, 2 * half_wid))

    bench = Bench(args, cfg, obstacles, half_len, half_wid)
    if not bench.preflight():
        rospy.logfatal("[bench] pre-flight failed; not running any trial")
        return 2

    try:
        results = bench.run()
    except rospy.ROSInterruptException:
        log("interrupted")
        return 1

    # Hash the world file this run actually had in front of it, rather than
    # trusting the goal set's `meta.world_sha1`.  Those are two different
    # claims and they can disagree: the goal set is a static file that records
    # what the world hashed to when gen_goals.py ran, so regenerating it after a
    # run has started leaves that run reporting a hash for a world it never
    # loaded.  report.py then refuses the comparison -- or worse, passes it --
    # on the strength of metadata rather than on the thing being compared.
    # This happened for real: the physics fix landed, the cspace run started a
    # minute before the goal set was regenerated, and the two runs were
    # declared incomparable while having used byte-identical worlds.
    actual_world_sha1 = sha1_of(args.world)
    claimed = args.goals_doc["meta"].get("world_sha1")
    if claimed and claimed != actual_world_sha1:
        rospy.logwarn(
            "[bench] the goal set claims world_sha1 %s but %s hashes to %s.  "
            "Recording the file's actual hash; the goal set is stale.  If the "
            "goals were generated against the world this run is using, the "
            "coordinates are still valid -- only the recorded hash was wrong."
            % (claimed[:12], os.path.basename(args.world),
               actual_world_sha1[:12]))
    out = {
        "meta": {
            "planner": args.planner,
            "world": os.path.abspath(args.world),
            # The hash of the file this run was actually scored against.
            "world_sha1": actual_world_sha1,
            # What the goal set believes the world hashes to, kept so a
            # disagreement survives in the artefact and not just in a log line.
            "goal_set_world_sha1": claimed,
            "goal_set": os.path.abspath(args.goals),
            "goal_set_seed": args.goals_doc["meta"].get("seed"),
            "start": list(args.start),
            "robot": {"half_length": half_len, "half_width": half_wid},
            "bench_config": os.path.abspath(args.config),
            "goal_thresholds": cfg["goal"],
            "trial": cfg["trial"],
            "n_goals": len(results),
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
