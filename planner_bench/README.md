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
PLANNER_BENCH_GUI=false ./planner_bench/run.sh both
```

Use `cspace` or `neupan` in place of `both` for one planner. For a quick wiring
check, run `PLANNER_BENCH_GUI=false ./planner_bench/run.sh both --only 0 --timeout 12`.
The default opens Gazebo's GUI, matching the previous driver. Set
`PLANNER_BENCH_GUI=false` for headless runs. The script uses ROS port 11511 and
Gazebo port 11545 to avoid interfering with another session; both can be
overridden by `PLANNER_BENCH_ROS_PORT` and `PLANNER_BENCH_GAZEBO_PORT`.

`run.sh` starts the maintained `multi_robot_scenario` bringup with this copied
world, loads one planner through `start_planner.sh`, runs `bench.py`, then stops
only the processes it started. For `both`, it starts a fresh simulation for each
planner and calls `report.py` on the two JSON files. Reports land in `reports/`
or `PLANNER_BENCH_REPORT_DIR`. Previous files are retained with `.prev`.

## Benchmark definition

- `worlds/limo_complex.world` is a byte-for-byte copy of the currently used
  multi_robot_scenario world. `bench_common.py --selftest` checks its 20 moving
  models and 0.01 s physics step.
- The targets are hand-written in `config/bench.yaml` under `goals:` — one trial
  per entry, in the order written, each as `{x, y, yaw?}` or `[x, y, yaw?]`.
  Both planners run the same list from the same start, `(-3, 0, 0)`, outside the
  population spawn square. Entries are checked against `arena.bounds` and the
  world's static obstacle boxes at startup, so a typo fails before Gazebo
  starts; whether a target is *hard* is the author's call. Point coordinates are
  hashed into the run report, and `report.py` refuses to compare two runs whose
  targets differ (`--force` to override).
- `bench.py` checks `/clock`, 20 moving model poses, actual obstacle motion,
  `/odom`, `/limo/scan`, planner subscription, and required TF transforms before
  running. Every trial teleports the Limo, publishes its goal, and records the
  ground-truth trajectory from `/gazebo/model_states`.
- Arrival means remaining within 0.30 m of the final target while stopped for
  0.5 s. A trial has a 90 s simulation-time limit. The result includes path
  length, arrival time, final error, clearance, and achieved versus commanded
  speed. `report.py` checks the world and goal-file hashes before comparing.

The obstacle trajectories are **not identical between planner runs**: Gazebo's
population placement and RVO goal assignment are random. The report compares
the same targets under separate dynamic realizations, so a small difference in
success rate is not a controlled paired result. Run several repetitions before
drawing a performance conclusion.

Collision scoring uses Gazebo model poses and simple bounding circles for the
moving obstacles. This is a proximity proxy, especially for the human mesh;
it is not a Gazebo contact event. `min_clearance` and `collision` in the JSON
share that definition. The copied world has no fixed walls, so every target sits
in open static space and all difficulty comes from the moving obstacles — which
are re-randomized every run, and are the reason repetitions are needed.

## Files

`run.sh` orchestrates both runs, `start_planner.sh` sets private ROS parameters,
`bench.py` scores episodes, `bench_common.py` handles geometry and reads the
targets out of `config/bench.yaml`, and `report.py` compares JSON results.
Config and world data are kept beside them so the scripts can resolve paths from
their own location.
