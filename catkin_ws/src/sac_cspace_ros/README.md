# sac_cspace_ros

ROS 1 (Noetic) wrapper for the **CNN-SAC c-space navigation policy** trained in
`DRL-robot-navigation/SAC` — the cost-ray actor from `Sparrow_V2.SparrowCspaceEnv`.

The node rebuilds the training observation from live sensor data and publishes
velocity commands:

```
/r1/front_laser/scan ─┐
/r1/odom ─────────────┼─→ occ grid → c-space → 36×nz cost rays → log encode ─┐
/sac_cspace/goal ─────┘                                                       ├─→ actor → /r1/cmd_vel
                                    [distance, heading] ──────────────────────┘
```

It calls the same pure-numpy/torch helpers the training env uses
(`SAC/cspace_cost_rays.py: make_footprint_masks / build_cspace / cost_rays_3d /
preprocess_cost_rays`), so **no Gazebo and no C++ `c_space_mapper` node are
required** — just a laser scan, odometry and a tf tree.

> ### ⚠ Breaking change: the kinematic block is now `[distance, heading]`
>
> The observation used to append the previous action command on top of
> `[distance, heading]`, making `state` 1084-dim. That action feedback was removed
> from the training env, so the state is now **1082-dim** (`36×30 + 2`).
>
> **Every checkpoint currently in `SAC/pytorch_models/` predates this change and
> can no longer be loaded.** `check_models.py` reports `0/11 ok`, and the node
> refuses to start with a message naming the offending model and saying it must be
> retrained. That is the intended behaviour, not a regression — retrain with
> `SAC/train.py` (it reads `settings.KINEMATIC_DIM`, so nothing else needs
> editing) and the config profiles below apply to the new checkpoint as-is.

## Quick start

```bash
cd catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5
source devel/setup.bash

# validate every checkpoint against its config first (no ROS master needed)
rosrun sac_cspace_ros check_models.py

# terminal 1: the robot (existing scenario)
roslaunch multi_robot_scenario multi_robot_scenario.launch
# terminal 2: the policy
roslaunch sac_cspace_ros sac_cspace.launch
# terminal 3: send a goal
rosrun sac_cspace_ros send_goal.py 3.0 3.0
```

`~sac_dir` defaults to `/DRL-robot-navigation/SAC`; set it if your checkout lives
elsewhere. `~device` defaults to `cuda` in the configs — `cpu` is plenty for a
single 36×nz grid.

## Topics

| direction | topic | type | notes |
|---|---|---|---|
| in | `/r1/front_laser/scan` | `sensor_msgs/LaserScan` | `~scan_topic` |
| in | `/r1/odom` | `nav_msgs/Odometry` | `~odom_topic` |
| in | `/sac_cspace/goal` | `geometry_msgs/PoseStamped` | `~goal_topic`; latch-free, re-send to retarget |
| out | `/r1/cmd_vel` | `geometry_msgs/Twist` | `~cmd_vel_topic` |
| out | `/sac_cspace/goal_marker` | `visualization_msgs/Marker` | goal cylinder, `~publish_debug` |
| out | `/sac_cspace/cost_ray_path` | `nav_msgs/Path` | θ=0 cost-ray polar profile — the perceived free space, for rviz |

## Config profiles — pick the one matching your checkpoint

`SAC/pytorch_models/` holds three checkpoint families that are **not**
interchangeable. The network architecture is sniffed from the state dict
automatically, but the **geometry and the ray pipeline are not recoverable from a
checkpoint**, so they come from a config file. The node validates the geometry
against the loaded network and **aborts on a mismatch** rather than navigating with
a wrong observation (which would be silent and dangerous).

| config | use with | ray grid | features | pipeline |
|---|---|---|---|---|
| `config/sac_cspace_sparrow.yaml` | `dense_sample_cspace_{0,1,2}` | 36×30 | 64 | log encode, `env` frame, 0.5 m/s / 2 rad/s, 3 m, 0.1 m cells, 1.20×0.14 m footprint |
| `config/sac_cspace_legacy_sparrow.yaml` | `SAC_sparrow_cspace{,_1..4}` | 36×10 | 128 | same, but `a=0.698` → nz=10 |
| `config/sac_cspace_legacy_gazebo.yaml` | `SAC_cspace{,_1,_2}` | 36×10 | 128 | **experimental** — subtract encode, `ros` frame, 1.0 m/s / 1.0 rad/s, 5 m, 0.2 m cells, 0.44×0.38 m footprint |

```bash
roslaunch sac_cspace_ros sac_cspace.launch \
    config:=$(rospack find sac_cspace_ros)/config/sac_cspace_legacy_sparrow.yaml \
    model_name:=SAC_sparrow_cspace_2
```

> `SAC_cspace_*` and `SAC_sparrow_cspace_*` have byte-identical network shapes, so
> only the config distinguishes their two different ray pipelines. The
> `legacy_gazebo` profile is transcribed from
> `catkin_ws/src/c_space_mapper/config/mapper_node.yaml` plus the training code,
> but its `ray_frame` and velocity scale are *inferred*, not verified against a
> rollout — check the behaviour before trusting it.

## Two details that are easy to get wrong

These are load-bearing; getting either wrong yields a plausible-looking but wrong
policy, which is why they are explicit config knobs rather than hardcoded.

1. **Grid handedness.** `SparrowCspaceEnv._build_occ_grid` projects hits with
   `rel_y = dx·sin(t) + dy·cos(t)`, so the occupancy grid's second axis runs to the
   robot's **right** — the mirror of the standard ROS body frame. Feed a ROS-frame
   grid directly and every ray comes out left/right swapped (differences up to the
   full 3 m range). `ray_frame: env` negates the ROS left component before
   quantisation; `ray_frame: ros` is the un-mirrored variant the old C++ mapper
   produced.

2. **The kinematic block is `[distance, heading]` only.** The observation used to
   also append the previous action command (`v_lin, v_ang`), making it
   `[dist, heading, v_lin, v_ang]`; that feedback was removed from the training env
   and no longer exists here. The consequence: **checkpoints trained before the
   change carry a `feature_dim + 4` first actor layer and cannot load** — the node
   detects this and tells you to retrain rather than failing with a shape mismatch.

Also worth knowing: the cost rays are quantised to 0.05 m march steps on a 0.1 m
grid, so raw values can reach ~3.05 m before the log encoding clips them at
`max_range`; and the sim's lidar is quantised to 8 cm and only ray *endpoints* are
marked occupied (there is no ray-traced clearing). A real laser gives a richer grid
— that is the intended sim→real direction, not a bug.

## Parameters

| param | default | meaning |
|---|---|---|
| `~model_name` | `dense_sample_cspace_2` | checkpoint basename in `~model_dir` |
| `~model_dir` | `<sac_dir>/pytorch_models` | where the `*_actor.pth` files live |
| `~sac_dir` | `/DRL-robot-navigation/SAC` | training tree, for `cspace_cost_rays` |
| `~device` | `cuda` | `cuda` or `cpu` |
| `~stochastic` | `false` | `true` samples from the policy instead of `tanh(mean)` |
| `~g_min`, `~nz`, `~a` | `0.1`, `30`, `2π/30` | c-space cell size, θ layers, θ step |
| `~cost_ray_max_range` | `3.0` | ray horizon (m) |
| `~cost_ray_num_beams` | `36` | beams per layer |
| `~cost_ray_h_res_degrees` | `10.0` | beam spacing |
| `~footprint` | `[[0.6,0.07],…]` | robot polygon, metres, +x forward |
| `~ray_frame` | `env` | `env` \| `ros` — see above |
| `~preprocess` | `log` | `log` \| `subtract` |
| `~v_linear_max`, `~v_angular_max` | `0.5`, `2.0` | action scale (m/s, rad/s) |
| `~allow_reverse` | `true` | maps `a0` to `[0,1]` when false |
| `~rate` | `10.0` | control rate (Hz); training used 0.1 s |
| `~goal_tolerance` | `0.3` | arrival radius (m); training's `target_area` |
| `~scan_max_range` | `0` | hits beyond this count as no-return; `0` ⇒ use `~cost_ray_max_range` |
| `~scan_min_range` | `0` | ignore returns closer than this |
| `~publish_debug` | `true` | publish the goal marker and cost-ray path |

## Tools

- `check_models.py` — load every checkpoint against a config and run one
  observation→action step; catches both a config/checkpoint geometry mismatch and
  a checkpoint too old to load (wrong kinematic block), before a rollout.
  ```bash
  rosrun sac_cspace_ros check_models.py --config config/sac_cspace_legacy_sparrow.yaml
  ```
  It shares `_kin_dim_of` and `KINEMATIC_DIM` with the node, so it cannot report a
  model as loadable that the node would reject.
- `send_goal.py` — publish a goal `PoseStamped`.
  ```bash
  rosrun sac_cspace_ros send_goal.py 3.0 3.0 --frame odom
  ```

## Caveats

- **Single goal, no global planner.** The shipped checkpoints were trained with
  `settings.GLOBAL_PLAN_ENABLED = False`, i.e. against a randomly placed target
  point, so the node drives straight at one goal and stops within
  `~goal_tolerance`. Feeding it graph-planner waypoints is a shift in the target
  distribution — the policy only sees distance/heading to the current waypoint, so
  it should transfer, but the waypoint statistics would not match training.
- **No collision checking or safety layer.** The actor is the only thing between
  the robot and an obstacle. Add a watchdog before running on hardware.
- The old Gazebo family is loaded but its ray pipeline is inferred, not verified
  (see above).
- `SAC/play_cspace.py:118` hardcodes `cnn_feature_dim=64`, so it cannot load the
  `SAC_sparrow_cspace_*` checkpoints (which need 128). That is a bug in that
  script, not here — this node reads the feature width from the state dict.
