#!/usr/bin/env python3

"""Run the shared benchmark: start the simulation, load one planner, score it.

    python3 run_bench.py cspace
    python3 run_bench.py both --only 0 --timeout 12
    PLANNER_BENCH_GUI=false python3 run_bench.py both

This is the main program.  Everything planner-specific lives in
`start_planner.sh`, everything scoring-specific in `bench.py`, and everything
comparison-specific in `report.py`; this file only owns the simulation
lifecycle -- what to start, in what order, and what to kill afterwards.

It is Python rather than the shell script it replaces so that the arena's start
pose and the trial plan come from the same places `bench.py` reads them from
(the world config and `bench_common.trial_plan`), instead of being extracted
from YAML by an inline one-liner that could disagree with the harness.

Environment (same names as the shell script, so existing invocations keep
working): PLANNER_BENCH_GUI, PLANNER_BENCH_ROS_PORT, PLANNER_BENCH_GAZEBO_PORT,
PLANNER_BENCH_WORLD, PLANNER_BENCH_WORLD_CONFIG, PLANNER_BENCH_CONFIG,
PLANNER_BENCH_REPORT_DIR, PLANNER_BENCH_LOG_DIR, and the two planner overrides
read by start_planner.sh.
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, HERE)
from bench_common import (load_world_config, trial_plan, goals_from_config,
                          parse_world_obstacles, world_config_path)


def env_path(name, default):
    return os.environ.get(name, default)


def report_dir():
    return env_path("PLANNER_BENCH_REPORT_DIR", os.path.join(HERE, "reports"))


def log_dir():
    return env_path("PLANNER_BENCH_LOG_DIR",
                    os.path.join(os.environ.get("TMPDIR", "/tmp"),
                                 "planner_bench"))


def say(msg):
    sys.stdout.write("[run] %s\n" % msg)
    sys.stdout.flush()


class Session(object):
    """The processes this script started, and the promise to kill only those.

    Both the simulation and the planner are started with setsid, so each is its
    own process group, and the group -- not the leader -- is signalled, because
    roslaunch's own children share its group until they leave it.

    The group id is captured at spawn rather than looked up at kill time.  With
    setsid the child is its own group leader, so pgid == pid -- and that is the
    only moment the mapping is reliable.  Once the leader is reaped
    ``os.getpgid`` raises, so a lookup at kill time finds nothing to signal.

    This is necessary but NOT sufficient, which is easy to get wrong: roslaunch
    starts rosmaster and gzserver, and both move themselves into a *new session*
    (pgid == their own pid, parent becomes init) as they start.  They therefore
    leave this group, and killing it takes roslaunch down and leaves them alive
    holding the ROS master port -- which is what makes the next run refuse to
    start.  ``reap_run()`` finishes that job by command line; see its docstring.
    """

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.sim = None
        self.planner = None

    def _spawn(self, argv, log_name):
        log_path = os.path.join(self.log_dir, log_name)
        fh = open(log_path, "w")
        p = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL,
                             preexec_fn=os.setsid)
        p._log = log_path
        p._fh = fh
        p._pgid = p.pid                          # setsid makes it the leader
        return p

    @staticmethod
    def _signal_group(pgid, sig):
        """Signal every member of ``pgid``.  False if the group is empty."""
        try:
            os.killpg(pgid, sig)
            return True
        except OSError:
            return False                         # no members left

    @staticmethod
    def _group_alive(pgid):
        """Is anyone in ``pgid`` still running?

        Not ``killpg(pgid, 0)``: that succeeds for a group whose members have
        all exited but not yet been reaped, and this runs *before* ``p.wait()``,
        so the just-killed leader is still a zombie and every teardown would
        report a phantom survivor.  A zombie holds no port and cannot block the
        next run, so it does not count as alive.
        """
        for pid, argv in iter_cmdlines():
            try:
                with open("/proc/%d/stat" % pid, "rb") as fh:
                    stat = fh.read().decode("utf-8", "replace")
            except (OSError, IOError):
                continue
            rest = stat.rsplit(")", 1)[-1].split()   # comm may contain spaces
            if len(rest) < 3:
                continue
            state, pgrp = rest[0], rest[2]
            if state != "Z" and pgrp.isdigit() and int(pgrp) == pgid:
                return True
        return False

    def _stop(self, p):
        if p is None:
            return
        pgid = getattr(p, "_pgid", None) or os.getpgid(p.pid)
        self._signal_group(pgid, signal.SIGTERM)
        for _ in range(50):                      # 5 s to go quietly
            if not self._group_alive(pgid):
                break
            time.sleep(0.1)
        if self._group_alive(pgid):              # still someone running
            self._signal_group(pgid, signal.SIGKILL)
            for _ in range(20):                  # 2 s for the group to drain
                if not self._group_alive(pgid):
                    break
                time.sleep(0.1)
            if self._group_alive(pgid):
                print("[run] warning: pgid %d still alive after SIGKILL" % pgid,
                      file=sys.stderr)
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        p._fh.close()

    def stop_planner(self):
        self._stop(self.planner)
        self.planner = None

    def stop_sim(self):
        self._stop(self.sim)
        self.sim = None

    def stop_all(self):
        # Planner first: it is the one that reacts to a live master, and killing
        # the master from under it produces a page of ROS errors in its log that
        # is not a planner fault.
        self.stop_planner()
        self.stop_sim()


ROS_TOOLS = ("roslaunch", "rosrun", "rosparam", "rosservice")


def iter_cmdlines():
    """(pid, argv) for every process we can read, via /proc.  No external tool:
    ``ss`` is not installed in the bench container, so a port-based probe would
    silently find nothing and report success."""
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open("/proc/%s/cmdline" % entry, "rb") as fh:
                argv = fh.read().decode("utf-8", "replace").split("\0")
        except (OSError, IOError):
            continue
        argv = [a for a in argv if a]
        if argv:
            yield int(entry), argv


def orphan_pids(ros_port, world):
    """PIDs of a dead run's leftovers, matched on their command lines.

    A process group is not enough to reap a ROS run.  roslaunch starts rosmaster
    and gzserver, but both put themselves in a *new session* (pgid == own pid,
    parent becomes init), so they leave roslaunch's group as they start.  Killing
    that group takes roslaunch down and leaves those two alive -- which is the
    orphan that then blocks the next run with "master already running".

    Matched on the *basename* of each argv token, not the raw token: roslaunch
    spawns these by absolute path (``/opt/ros/noetic/bin/rosmaster``) or through
    a shell (``/bin/sh /opt/ros/noetic/lib/gazebo_ros/gzserver``), and only the
    grandchild appears as a bare ``gzserver``.  The second condition is still an
    exact argv token -- this run's port for the master, this run's world file for
    Gazebo -- so a process that merely mentions ROS cannot be caught.
    """
    pids = []
    for pid, argv in iter_cmdlines():
        if pid == os.getpid():
            continue
        names = set(os.path.basename(a) for a in argv)
        if "rosmaster" in names and ros_port in argv:
            pids.append(pid)
        elif names & {"gzserver", "gzclient"} and world in argv:
            pids.append(pid)
    return pids


def reap_run(ros_port, world):
    """Kill what the group teardown could not reach.  Returns the PIDs killed."""
    pids = orphan_pids(ros_port, world)
    if not pids:
        return []
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except OSError:
                pass
        for _ in range(50):
            if not orphan_pids(ros_port, world):
                return pids
            time.sleep(0.1)
    left = orphan_pids(ros_port, world)
    if left:
        print("[run] warning: could not reap %s" % left, file=sys.stderr)
    return pids


def missing_ros_tools():
    return [t for t in ROS_TOOLS if shutil.which(t) is None]


def run_one(mode, args, cfg_path, world, world_config, start):
    """One planner, one fresh simulation, scored into its own report file."""
    out = os.path.join(report_dir(), "%s_run.json" % mode)
    os.makedirs(report_dir(), exist_ok=True)
    os.makedirs(log_dir(), exist_ok=True)
    gui = env_path("PLANNER_BENCH_GUI", "true")
    ros_port = env_path("PLANNER_BENCH_ROS_PORT", "11511")

    session = Session(log_dir())
    # Any exit path -- return, exception, Ctrl-C, SIGTERM -- must leave no
    # orphaned Gazebo behind.  Returning from the handler rather than raising
    # keeps a Ctrl-C from printing a traceback and skipping the report step.
    def bail(signum, frame):
        session.stop_all()
        if signum is not None:
            sys.exit(128 + signum)

    old = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        old[sig] = signal.signal(sig, bail)
    try:
        # A master already on this port means an earlier run's Gazebo survived
        # (SIGKILL, a crash, a killed container), and this run would attach to it
        # and score the wrong world.  Leftovers from that run are necessarily
        # ours -- the port is dedicated to the bench -- so clear them and carry
        # on rather than making the operator hunt them down.
        stale = reap_run(ros_port, world)
        if stale:
            say("cleared %d leftover process(es) from a previous run: %s"
                % (len(stale), stale))
            time.sleep(1)
        if subprocess.call(["rosparam", "get", "/run_id"],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL) == 0:
            say("ROS master at %s is already running; choose another "
                "PLANNER_BENCH_ROS_PORT" % os.environ["ROS_MASTER_URI"])
            return 1
        if os.path.exists(out) and os.path.getsize(out) > 0:
            os.replace(out, out + ".prev")

        say("starting simulation: %s" % os.path.basename(world))
        session.sim = session._spawn(
            ["roslaunch", "-p", ros_port, "multi_robot_scenario",
             "multi_robot_scenario.launch", "world_name:=%s" % world,
             "obstacles:=true", "gui:=%s" % gui,
             "x:=%g" % start[0], "y:=%g" % start[1], "yaw:=%g" % start[2]],
            "sim.log")

        ready = False
        for _ in range(120):
            if session.sim.poll() is not None:
                say("simulation stopped; see %s" % session.sim._log)
                return 1
            try:
                svc = subprocess.check_output(["rosservice", "list"],
                                              stderr=subprocess.DEVNULL)
            except (OSError, subprocess.CalledProcessError):
                svc = b""
            if b"/gazebo/set_model_state" in svc.split():
                ready = True
                break
            time.sleep(1)
        if not ready:
            say("Gazebo did not become ready; see %s" % session.sim._log)
            return 1

        session.planner = session._spawn(
            [os.path.join(HERE, "start_planner.sh"), mode], "%s.log" % mode)
        time.sleep(2)
        if session.planner.poll() is not None:
            say("planner stopped; see %s" % session.planner._log)
            return 1

        say("running %s against %s" % (mode, os.path.basename(world)))
        rc = subprocess.call(
            [sys.executable, os.path.join(HERE, "bench.py"),
             "--planner", mode, "--world", world, "--config", cfg_path,
             "--world-config", world_config, "--out", out] + args.rest)
        if rc == 0:
            say("wrote %s" % out)
        return rc
    finally:
        session.stop_all()
        # The group teardown above cannot reach rosmaster/gzserver once they have
        # put themselves in their own sessions, so finish the job by command line.
        # This is what keeps the "master already running" guard from firing on the
        # next run.
        reaped = reap_run(ros_port, world)
        if reaped:
            say("reaped %d orphan(s) from the teardown: %s" % (len(reaped), reaped))
        for sig, handler in old.items():
            signal.signal(sig, handler)


def main(argv):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["cspace", "neupan", "both"])
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="arguments passed through to bench.py "
                         "(e.g. --only 0 --timeout 12)")
    args = ap.parse_args(argv)

    missing = missing_ros_tools()
    if missing:
        sys.exit("Source /opt/ros/noetic/setup.bash and "
                 "catkin_ws/devel/setup.bash first: %s not on PATH."
                 % ", ".join(missing))

    world = env_path("PLANNER_BENCH_WORLD",
                     os.path.join(HERE, "worlds", "limo_complex.world"))
    cfg_path = env_path("PLANNER_BENCH_CONFIG",
                        os.path.join(HERE, "config", "bench.yaml"))
    world_config = env_path("PLANNER_BENCH_WORLD_CONFIG",
                            world_config_path(world))

    # Read the start pose through the same loader bench.py uses.  The launch
    # file spawns the robot there and bench.py teleports to it before every
    # trial, so a disagreement between the two would show up as every trial
    # starting at the wrong place -- worth failing loudly over, before Gazebo.
    try:
        wcfg = load_world_config(world_config)
        start = list(wcfg["arena"]["start"])
    except (IOError, OSError) as exc:
        sys.exit("[run] %s" % exc)
    except (ValueError, KeyError) as exc:
        sys.exit("[run] %s: %s" % (world_config, exc))

    ros_port = env_path("PLANNER_BENCH_ROS_PORT", "11511")
    os.environ["ROS_MASTER_URI"] = "http://127.0.0.1:%s" % ros_port
    os.environ["GAZEBO_MASTER_URI"] = "http://127.0.0.1:%s" % env_path(
        "PLANNER_BENCH_GAZEBO_PORT", "11545")

    # Report the schedule before Gazebo comes up.  Ten trials over three targets
    # is a half-hour of simulation, and a typo in `trials:` or `goal_sample:` is
    # not worth discovering after the first episode.
    try:
        obstacles = parse_world_obstacles(world, margin=0.0)
        goals = goals_from_config(wcfg, obstacles=obstacles,
                                  bounds=tuple(wcfg["arena"]["bounds"]),
                                  start=start)
        plan = trial_plan(goals, trials=wcfg.get("trials"),
                          sample=wcfg.get("goal_sample", "cycle"),
                          seed=wcfg.get("trial_seed"))
    except (IOError, OSError, ValueError) as exc:
        sys.exit("[run] %s: %s" % (world_config, exc))

    sample = wcfg.get("goal_sample", "cycle")
    say("%d target(s) -> %d trial(s), sample=%s, start=(%g, %g, %g)"
        % (len(goals), len(plan), sample, start[0], start[1], start[2]))
    for gi in sorted({t["goal_i"] for t in plan}):
        n = sum(1 for t in plan if t["goal_i"] == gi)
        say("  target %d (%.2f, %.2f)  %.2f m  x%d"
            % (gi, goals[gi]["x"], goals[gi]["y"], goals[gi]["distance"], n))

    modes = ["cspace", "neupan"] if args.mode == "both" else [args.mode]
    for mode in modes:
        rc = run_one(mode, args, cfg_path, world, world_config, start)
        if rc != 0:
            return rc

    if args.mode == "both":
        return subprocess.call(
            [sys.executable, os.path.join(HERE, "report.py"),
             os.path.join(report_dir(), "cspace_run.json"),
             os.path.join(report_dir(), "neupan_run.json")])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
