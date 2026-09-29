# NeuPAN on the thin "sparrow" footprint

Runs NeuPAN on the same Gazebo LIMO as `neupan_gazebo_limo.launch`, but planned
as the **1.20 x 0.14 m** body defined by `SAC/settings.py`
(`ROBOT_FOOTPRINT_LENGTH` / `ROBOT_FOOTPRINT_WIDTH`) — the footprint the CNN-SAC
c-space policy in `cnav_ros/config/cnav_sparrow.yaml` navigates with. The point of
the profile is to compare the two planners on identical geometry.

| file | what |
|---|---|
| `config/neupan_planner_sparrow.yaml` | planner profile: 1.2 x 0.14 m hull, 0.5 m/s, 2 rad/s |
| `launch/neupan_gazebo_sparrow.launch` | the node + rviz, remapped onto the LIMO |
| `pretrain_sparrow_thin/model_5000.pth` | DUNE for that hull, trained in `NeuPAN/example/dune_train/` (`name: sparrow_thin`) |

## Run it

```bash
roslaunch neupan_ros gazebo_limo_env_complex_20.launch
roslaunch neupan_ros neupan_gazebo_sparrow.launch
```

Then set a goal with rviz's **2D Nav Goal** tool (the launch remaps
`/neupan_goal` to `/move_base_simple/goal`).

## Three things that will bite you

### 1. `scan_range` must not start at 0

The LIMO laser clips the robot's own body on the beams at the ends of its FOV,
returning **0.05-0.09 m** (measured; recorded in `cnav_sparrow.yaml`, and the
0.05 floor is the Gazebo sensor's own `<min>`). Those points land *inside* the
1.2 x 0.14 m hull, so DUNE scores them as a near-zero or negative distance and
`min_distance < collision_threshold` makes the planner declare a collision and
stop on the spot before it has moved.

Measured in the headless planner-in-the-loop harness (`scan_min` = the launch's
`scan_range` minimum, `open` = empty world, goal 5 m ahead):

| `scan_min` | result |
|---|---|
| `0.0` | robot never moves: `mean abs(v) = 0.000 m/s`, `stop` on 100% of steps |
| `0.1` | drives off normally at 0.5 m/s and reaches the goal |

`0.1`, not `0.09`: the filter keeps `distance >= scan_range[0]`, so a surviving
0.09 m return still lands ~8 mm outside the hull, under `collision_threshold`
(0.01). Filtering by range rather than by beam index also makes it independent of
where the lidar is mounted.

### 2. The hull and the checkpoint must agree

DUNE encodes the robot's convex hull. `robot.length`/`robot.width` in the config
and the checkpoint in the launch must describe the same polygon or the planner is
solving for a body it does not have. To change the body, edit `length`/`width`
(and `name`), retrain with `NeuPAN/example/dune_train/dune_train_sparrow.yaml`,
and point the launch at the new checkpoint. Note the existing `pretrain_limo/`
checkpoint does **not** work with this config, nor this one with the limo config.

### 3. Re-send the goal if the robot wanders

`initial_path` is tracked by `closest_point`, which only searches **forward**
along the path and never revisits an earlier index. A deviation the robot cannot
undo on its own (a lateral offset of more than roughly the `arrive_threshold`,
0.3 m) leaves it tracking a reference that is now behind it. Re-sending the goal
fixes it: `goal_callback` reruns `update_initial_path_from_goal(robot_state, goal)`
and `reset()`, drawing a fresh line from where the robot actually is.

## What was verified

The DUNE itself is checked against a direct ECOS solve of the same distance
programme in `NeuPAN/example/dune_train/README_sparrow_thin.md` (near field MAE
0.4 mm). This profile was then checked two ways, both without ROS:

- **Construction path** — the planner is built exactly as `neupan_core.py` builds
  it (`init_from_yaml(cfg, pan={'dune_checkpoint': ckpt})`) and stepped once. Note
  that this call *replaces* the yaml's whole `pan` section rather than merging, so
  the yaml's `iter_num` / `dune_max_num` / `nrmp_max_num` / `iter_threshold` are
  inert; they happen to equal the library defaults today, so nothing breaks, but
  editing them in the config will have no effect. The `adjust` section is a
  separate key and *is* honoured.
- **Closed loop** — a unicycle model of the 1.2 x 0.14 m body, a simulated 720-beam
  240 deg lidar matching `limo_gazebo.gazebo` (8 m range, cut to `scan_range`),
  replanning at the node's 50 Hz, with clearance to the world geometry measured
  exactly (rotated rectangle vs. axis-aligned boxes), never through DUNE.

<!-- measurement tables pending -->
