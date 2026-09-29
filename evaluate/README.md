# Planner benchmark scripts

This directory runs cspace and NeuPAN against the same Limo robot and a copied
`multi_robot_scenario/worlds/limo_complex.world`. It is a script directory, not a
catkin package. The world contains 20 moving obstacles. Their models and RVO
driver are provided by `multi_robot_scenario`, so that ROS package must remain
in the sourced workspace.

## Run

Inside the Noetic container:

```bash
source /opt/ros/noetic/setup.bash
source /DRL-robot-navigation/catkin_ws/devel/setup.bash
cd /DRL-robot-navigation
PLANNER_BENCH_GUI=false ./evaluate/run_bench.py both
```

Use `cspace` or `neupan` in place of `both` for one planner. For a quick wiring
check, run `PLANNER_BENCH_GUI=false ./evaluate/run_bench.py both --only 0 --timeout 12`.
The default opens Gazebo's GUI. Set `PLANNER_BENCH_GUI=false` for headless runs.
The runner uses ROS port 11511 and Gazebo port 11545 to avoid interfering with
another session; both can be overridden by `PLANNER_BENCH_ROS_PORT` and
`PLANNER_BENCH_GAZEBO_PORT`. The same environment variables the old `run.sh`
read still work, including `PLANNER_BENCH_WORLD`, `PLANNER_BENCH_WORLD_CONFIG`,
`PLANNER_BENCH_CONFIG`, `PLANNER_BENCH_REPORT_DIR` and `PLANNER_BENCH_LOG_DIR`.

`run_bench.py` is the main program. It starts the maintained
`multi_robot_scenario` bringup with this copied world, loads one planner through
`start_planner.sh`, runs `bench.py`, then stops only the processes it started.
Both are launched with `setsid` so each gets its own process group, but a group
kill is *not* sufficient on its own: `roslaunch` starts `rosmaster` and
`gzserver`, and both move themselves into a new session as they start, so they
leave roslaunch's group and survive the kill holding the master port. What makes
that visible is the next run refusing to start. So after the group teardown the
runner reaps by command line, matching this run's `-p` port for the master and
this run's world file for Gazebo, and it does the same sweep *before* starting —
a leftover from a crashed run is cleared rather than handed back as an error.
For `both`, it starts a fresh simulation for each planner and calls `report.py`
on the two JSON files. Reports land in `reports/` or `PLANNER_BENCH_REPORT_DIR`.
Previous files are retained with `.prev`.

Before Gazebo starts it expands the trial plan from the world config and prints
it, so a typo in `trials:` or a target outside `arena.bounds` costs a second
rather than a failed episode:

```
[run] 3 target(s) -> 10 trial(s), sample=cycle, start=(-3, 0, 0)
[run]   target 0 (5.00, 0.00)  8.00 m  x4
[run]   target 1 (0.00, 4.00)  5.00 m  x3
[run]   target 2 (2.00, -3.00) 5.83 m  x3
```

## Benchmark definition

- `worlds/limo_complex.world` is a copy of the currently used
  multi_robot_scenario world. Gazebo supplies moving model poses during a run.
- Everything that changes with the map lives in `worlds/<stem>.yaml`, beside the
  world it describes (`worlds/limo_complex.yaml` for
  `worlds/limo_complex.world`). `bench.py` derives that path from `--world`, so
  a run cannot use one map's bounds with another map's targets;
  `--world-config` overrides it for a world kept elsewhere. It holds `arena`
  (bounds, start, start_clearance), the hand-written `goals:` (each as
  `{x, y, yaw?}` or `[x, y, yaw?]`), plus
  `trials:` and `goal_sample:`. Both planners run the same targets from the same
  start, `(-3, 0, 0)`, outside the population spawn square. Entries are checked
  against `arena.bounds` at startup; obstacle clearance and target difficulty are
  the author's choice. Point coordinates are hashed into the run report, and
  `report.py` refuses to compare two runs whose targets differ (`--force` to
  override).
- **Trials draw targets at random.** `trials: 10` is the number of Gazebo
  episodes, not the number of targets; each episode picks one target from
  `goals:`. `goal_sample: cycle` walks a fresh random permutation each time it
  runs out, so every target is used the same number of times to within one and
  only the order is random; `random` picks uniformly with replacement, which at
  low trial counts can leave a target unused. `trial_seed:` fixes the schedule
  so the same sequence of trials can be reproduced — it does not fix the run,
  since Gazebo re-randomizes the moving obstacles either way. Omit `trials:` to
  run each target exactly once.
- `config/bench.yaml` holds what is true of the harness on any map: the robot
  footprint polygon and the body polygon that masks its own laser returns, the
  trial timeouts, the success radius, and the laser margin the collision test
  uses.
- `bench.py` checks `/clock`, 20 moving model poses, actual obstacle motion,
  `/odom`, `/limo/scan`, planner subscription, and required TF transforms before
  running. Every trial teleports the Limo, publishes its goal, and records the
  ground-truth trajectory from `/gazebo/model_states`.
- Arrival means remaining within 0.30 m of the final target while stopped for
  0.5 s. A trial has a 90 s simulation-time limit. The result includes path
  length, arrival time, final error, scan margin, and achieved versus commanded
  speed. `report.py` checks the world, target-set, start, footprint and body-mask
  hashes before comparing, and refuses if the two runs did not exercise the same
  targets, were not scored against the same footprint, or were not scored with
  the same self-return mask.

The obstacle trajectories are **not identical between planner runs**: Gazebo's
population placement and RVO goal assignment are random, and each trial draws
its target at random. The report therefore groups by target, not by trial — two
runs' trial 3 are unrelated episodes — and gives `k/n` per target over the
trials that happened to draw it. Those trials are not paired with the other
planner's either, since the obstacles are re-randomized per trial. Run several
repetitions before drawing a performance conclusion.

Collision is read **directly off the laser**, against the robot's footprint
polygon. The footprint and the beam bearings are both in the robot frame, so for
each beam the only question is whether the return came back shorter than the
footprint's own radius in that direction: a shorter one terminated *inside* the
robot. No world pose, no odometry and no obstacle list take part.

A beam is dropped first when its return lands inside the **physical chassis**,
which is carried as a second polygon (`robot.body` in `config/bench.yaml`). This
is not tidiness. The scored footprint is 0.140 m wide and the real robot 0.220 m,
so the robot sticks out past its own silhouette and the laser sees its own body
sitting just *outside* it on nearly every bearing — a centimetre or so of margin.
Left in, that artefact is the smallest margin on almost every scan and decides
the verdict instead of traffic, and the first slightly-closer self-return ends
the trial at t=0. Measuring it: the minimum margin on each of 485 samples of one
real run came from a beam parked at ±119°, returning 0.090–0.099 m — the edge of
the sweep, where the robot's own chassis is. With the mask, the same trial's
minimum margin moved to a genuine obstacle at 0.25 m abeam and rose from 0.011 m
to 0.170 m.

The body polygon is the union of the URDF's collision primitives (wheels
included), 0.29 × 0.22 m, inflated by 2 cm to absorb the laser's own 1 cm range
noise — without that it has only ~5 mm of headroom over the self-returns actually
measured, and dead ahead it sits at 0.042 m, exactly the chassis front. The two
costs are that this is a line-of-sight test: it cannot see behind the robot (its
240° sweep), and abeam an obstacle closer than 0.130 m is masked as the robot
while the scored envelope only starts at 0.070 m — a 6 cm blind shell. Ahead,
nothing that could change a verdict is hidden: masked returns are under 0.062 m
while the envelope fires at 0.497 m. `python3 bench_common.py --footprint` prints
both silhouettes per bearing, which is the quickest way to see the trade.

The footprint is `SAC/settings.py`'s `ROBOT_FOOTPRINT` (1.20 × 0.14 m), carried
as a vertex list rather than a length × width box. It is cspace's *observation*
footprint — the c-space dilation its cost rays are computed against — and it is
neither the physical chassis (0.13 × 0.12 m; the `cspace_footprint` marker in
`limo_four_diff.xacro` draws both for comparison) nor what ended episodes during
training (a 14 cm circle, `SAC/Sparrow_V2.py`). Scored against it, a "collision"
fires while roughly 0.5 m of clear space is still ahead, so the collision counts
mean envelope violations rather than contact. Both planners are scored with the
same polygon, so the comparison stays like-for-like; `python3 bench_common.py
--footprint` prints the per-bearing threshold it implies.

The copied world has no fixed walls, so every target sits in open static space
and all difficulty comes from the moving obstacles — which are re-randomized
every run, and are the reason repetitions are needed.

## Files

`run_bench.py` orchestrates both runs, `start_planner.sh` sets private ROS
parameters, `bench.py` scores episodes, `bench_common.py` handles geometry and
reads the targets and the trial plan out of the world config, and `report.py`
compares JSON results. Config and world data are kept beside them so the scripts
can resolve paths from their own location. `worlds/` holds one `<stem>.world` +
`<stem>.yaml` pair per map. The harness uses the YAML for the trial schedule and
Gazebo topics for live model positions.

`bench_common.trial_plan()` is the one place the trial schedule is decided, and
`run_bench.py` calls it before starting Gazebo for the same numbers `bench.py`
will use. Anything that needs to know which target a trial drew reads
`goal_i`.
