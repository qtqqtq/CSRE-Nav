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
PLANNER_BENCH_GUI=false ./planner_bench/run_bench.py both
```

Use `cspace` or `neupan` in place of `both` for one planner. For a quick wiring
check, run `PLANNER_BENCH_GUI=false ./planner_bench/run_bench.py both --only 0 --timeout 12`.
The default opens Gazebo's GUI. Set `PLANNER_BENCH_GUI=false` for headless runs.
The runner uses ROS port 11511 and Gazebo port 11545 to avoid interfering with
another session; both can be overridden by `PLANNER_BENCH_ROS_PORT` and
`PLANNER_BENCH_GAZEBO_PORT`. The same environment variables the old `run.sh`
read still work, including `PLANNER_BENCH_WORLD`, `PLANNER_BENCH_WORLD_CONFIG`,
`PLANNER_BENCH_CONFIG`, `PLANNER_BENCH_REPORT_DIR` and `PLANNER_BENCH_LOG_DIR`.

`run_bench.py` is the main program. It starts the maintained
`multi_robot_scenario` bringup with this copied world, loads one planner through
`start_planner.sh`, runs `bench.py`, then stops only the processes it started --
each in its own process group, so `roslaunch`'s `gzserver`/`rosmaster` go with
it. For `both`, it starts a fresh simulation for each planner and calls
`report.py` on the two JSON files. Reports land in `reports/` or
`PLANNER_BENCH_REPORT_DIR`. Previous files are retained with `.prev`.

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

- `worlds/limo_complex.world` is a byte-for-byte copy of the currently used
  multi_robot_scenario world. `bench_common.py --selftest` checks its 20 moving
  models and 0.01 s physics step.
- Everything that changes with the map lives in `worlds/<stem>.yaml`, beside the
  world it describes (`worlds/limo_complex.yaml` for
  `worlds/limo_complex.world`). `bench.py` derives that path from `--world`, so
  a run cannot use one map's bounds with another map's targets;
  `--world-config` overrides it for a world kept elsewhere. It holds `arena`
  (bounds, start, start_clearance), the `world:` block the self-test asserts,
  the hand-written `goals:` (each as `{x, y, yaw?}` or `[x, y, yaw?]`), plus
  `trials:` and `goal_sample:`. Both planners run the same targets from the same
  start, `(-3, 0, 0)`, outside the population spawn square. Entries are checked
  against `arena.bounds` and the world's static obstacle boxes at startup, so a
  typo fails before Gazebo starts; whether a target is *hard* is the author's
  call. Point coordinates are hashed into the run report, and `report.py` refuses
  to compare two runs whose targets differ (`--force` to override).
- **Trials draw targets at random.** `trials: 10` is the number of Gazebo
  episodes, not the number of targets; each episode picks one target from
  `goals:`. `goal_sample: cycle` walks a fresh random permutation each time it
  runs out, so every target is used the same number of times to within one and
  only the order is random; `random` picks uniformly with replacement, which at
  low trial counts can leave a target unused. `trial_seed:` fixes the schedule
  so the same sequence of trials can be reproduced — it does not fix the run,
  since Gazebo re-randomizes the moving obstacles either way. Omit `trials:` to
  run each target exactly once.
- `config/bench.yaml` holds what is true of the harness on any map: the fallback
  robot footprint, the trial timeouts, the success radius.
- `bench.py` checks `/clock`, 20 moving model poses, actual obstacle motion,
  `/odom`, `/limo/scan`, planner subscription, and required TF transforms before
  running. Every trial teleports the Limo, publishes its goal, and records the
  ground-truth trajectory from `/gazebo/model_states`.
- Arrival means remaining within 0.30 m of the final target while stopped for
  0.5 s. A trial has a 90 s simulation-time limit. The result includes path
  length, arrival time, final error, clearance, and achieved versus commanded
  speed. `report.py` checks the world, target-set and start hashes before
  comparing, and refuses if the two runs did not exercise the same targets.

The obstacle trajectories are **not identical between planner runs**: Gazebo's
population placement and RVO goal assignment are random, and each trial draws
its target at random. The report therefore groups by target, not by trial — two
runs' trial 3 are unrelated episodes — and gives `k/n` per target over the
trials that happened to draw it. Those trials are not paired with the other
planner's either, since the obstacles are re-randomized per trial. Run several
repetitions before drawing a performance conclusion.

Collision scoring uses Gazebo model poses and simple bounding circles for the
moving obstacles. This is a proximity proxy, especially for the human mesh;
it is not a Gazebo contact event. `min_clearance` and `collision` in the JSON
share that definition. The copied world has no fixed walls, so every target sits
in open static space and all difficulty comes from the moving obstacles — which
are re-randomized every run, and are the reason repetitions are needed.

## Files

`run_bench.py` orchestrates both runs, `start_planner.sh` sets private ROS
parameters, `bench.py` scores episodes, `bench_common.py` handles geometry and
reads the targets and the trial plan out of the world config, and `report.py`
compares JSON results. Config and world data are kept beside them so the scripts
can resolve paths from their own location. `worlds/` holds one `<stem>.world` +
`<stem>.yaml` pair per map, and `python3 bench_common.py --selftest` checks the
pair without Gazebo.

`bench_common.trial_plan()` is the one place the trial schedule is decided, and
`run_bench.py` calls it before starting Gazebo for the same numbers `bench.py`
will use. Anything that needs to know which target a trial drew reads
`goal_i`.
