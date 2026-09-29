# CNav — `cnav_ros`

ROS 1 (Noetic) wrapper for the **CNN-SAC c-space navigation policy** trained in
`DRL-robot-navigation/SAC` — the cost-ray actor from `Sparrow_V2.SparrowCspaceEnv`.

The node rebuilds the training observation from live sensor data and publishes
velocity commands:

```
/scan ────┐
/odom ────┼─→ occ grid → c-space → 1×36×nz first-entry rays ──┐
/cnav_goal┘                                                       ├─→ actor → velocity filter → /cnav_cmd_vel
             [distance, heading, actual v, actual w] ─────────────┘
```

It calls the same pure-numpy/torch helpers the training env uses
(`SAC/cspace_cost_rays.py: make_footprint_masks / build_cspace /
cost_ray_intervals_3d / preprocess_cost_ray_intervals`), so it needs a laser
scan, odometry and a tf tree.

The local lidar grid keeps observed-free cells along beam paths, occupied cells
at finite hit endpoints, and unknown cells outside observed paths. In
configuration space, obstacle overlap is a collision; an otherwise obstacle-
free pose is invalid only when its entire footprint is unknown. A footprint
with any known-free overlap remains valid even if part of it is unknown.

> ### Current model: one ray interval plus measured velocity
>
> The state is **1084-dimensional**: one `36×30` ray-distance channel plus
> `[distance, heading, normalized_actual_v, normalized_actual_w]`. Velocity comes
> from the filtered simulator state during training and odometry during deployment;
> it is not the previous action command.
>
> Older checkpoints are intentionally incompatible and must be retrained.

## Quick start

```bash
cd catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5
source devel/setup.bash

# validate the configured checkpoint first (no ROS master needed)
rosrun cnav_ros check_models.py

# terminal 1: the environment - Gazebo + the Limo + 20 moving obstacles
roslaunch multi_robot_scenario multi_robot_scenario.launch
# terminal 2: the policy
roslaunch cnav_ros cnav.launch
# terminal 3: send a goal
rosrun cnav_ros send_goal.py 3.0 3.0
```

`cnav_rviz.launch` is the same policy with rviz attached, for watching it and
placing the goal with the **2D Nav Goal** tool instead of `send_goal.py`:

```bash
roslaunch cnav_ros cnav_rviz.launch              # policy + rviz
roslaunch cnav_ros cnav_rviz.launch rviz:=false  # policy alone
```

It starts no Gazebo — terminal 1 above is still required, and must come first so
rviz picks up `/use_sim_time`.

The **2D Nav Goal** tool has to publish on `~goal_topic` (`/cnav_goal`), which is
what `rviz/cnav.rviz` sets it to. rviz's own default is
`/move_base_simple/goal`, and nothing here subscribes to that, so a config left
on the default makes the tool look broken rather than mispointed — this is the
first thing to check if clicking the map does nothing. In the GUI it lives at
left panel → **Tool Properties** → **2D Nav Goal** → **Topic**. The click is
read in the fixed frame (`odom`): the node ignores the message's
`header.frame_id` and takes its own pose from `/odom`.

**2D Pose Estimate** is deliberately left unwired; nothing here reads
`/initialpose`.

`~sac_dir` defaults to `/DRL-robot-navigation/SAC`; set it if your checkout lives
elsewhere. `~device` defaults to `cuda` in the configs — `cpu` is plenty for a
single 36×nz grid.

## Node API

The names below follow the `/neupan_*` convention used by `neupan_ros`, so the
two controllers read the same way in a launch file.

Only interfaces that exist are listed. NeuPAN publishes `/dune_point_markers`
and `/nrmp_point_markers` because it has two internal optimisers, and
`/neupan_initial_path` and `/neupan_ref_state` because it runs an MPC — CNav is a
single reactive policy with none of those, so there is no placeholder for them.

### Publishers

| topic | type | description |
|---|---|---|
| `/cnav_cmd_vel` | `geometry_msgs/Twist` | velocity command to the robot |
| `/cnav_cost_ray_path` | `nav_msgs/Path` | the θ=0 cost-ray polar profile — the free space the policy currently perceives. **Not a planned path**, which is why it is not called `/cnav_plan` |
| `/cnav_cost_ray_image` | `sensor_msgs/Image` | the full 36×nz cost-ray grid as an `rgb8` heatmap — the observation the actor actually consumes. See [Reading the cost-ray image](#reading-the-cost-ray-image) |
| `/cnav_goal_marker` | `visualization_msgs/Marker` | cylinder at the current goal, gated by `~publish_debug` |

Both debug topics publish for as long as the node has a pose, whether or not
there is a goal: from startup, while driving, and after arrival. A parked robot
is exactly when "what does the policy perceive right now" is worth looking at,
and a viewer opened before the first goal click should not sit on "No Image".
(They used to be silent in both of those windows — before the first goal, and
again on arrival.)

The goal marker is the exception: it is published from `goal_callback`, so it
appears only once a goal exists.

### Subscribers

| topic | type | description |
|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | laser scan |
| `/odom` | `nav_msgs/Odometry` | odometry; supplies the pose in `~map_frame` |
| `/cnav_goal` | `geometry_msgs/PoseStamped` | sets the goal. Not latched — re-send to retarget |

Every one of them is a **parameter** (`~cmd_vel_topic`, `~plan_topic`,
`~marker_topic`, `~scan_topic`, `~odom_topic`, `~goal_topic`), not a launch
remap. The `evaluate` scripts load a fixed profile for each run, so the
topic selection stays visible in its config file.

So the table above is the API the node *declares*, and the config profile
retargets it at a concrete robot — the Limo profile maps them to `/cmd_vel`,
`/odom` and `/limo/scan`. This is the same arrangement NeuPAN has: its
`/neupan_cmd_vel` is remapped to `/cmd_vel` at startup and does not
exist at runtime either.

## Config profiles — pick the one matching your checkpoint

`SAC/pytorch_models/` holds three checkpoint families that are **not**
interchangeable. The network architecture is sniffed from the state dict
automatically, but the **geometry and the ray pipeline are not recoverable from a
checkpoint**, so they come from a config file. The node validates the geometry
against the loaded network and **aborts on a mismatch** rather than navigating with
a wrong observation (which would be silent and dangerous).

| config | use with | ray grid | features | pipeline |
|---|---|---|---|---|
| `config/cnav_sparrow.yaml` | `interval_1_2000000steps` in `pytorch_models/interval_sweep_2m` | 1×36×30 | 64 | normalized first-entry distance, measured velocity, `env` frame, 0.5 m/s / 2 rad/s, 3 m, 0.1 m cells, 1.20×0.14 m footprint |
| legacy profiles | old one-channel checkpoints | incompatible | — | retained only as historical geometry references; retraining is required |

```bash
roslaunch cnav_ros cnav.launch \
    config:=$(rospack find cnav_ros)/config/cnav_legacy_sparrow.yaml \
    model_name:=SAC_sparrow_cspace_2
```

> The old one-channel checkpoints used a different observation shape and need
> retraining for the current pipeline.

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

2. **Velocity means measured motion, not the previous command.** Training divides
   Sparrow's post-filter `car_state` velocity by `(v_linear_max, v_angular_max)`;
   deployment applies the same normalization to odometry twist. Both values are
   clipped to `[-1,1]`. Checkpoints without this complete layout are rejected.

Also worth knowing: the cost rays are quantised to 0.05 m march steps on a 0.1 m
grid and clipped to `max_range` before normalization; the sim's lidar is quantised
to 8 cm, with hit endpoints occupied and sampled beam paths marked known-free.
A real laser gives a richer grid
— that is the intended sim→real direction, not a bug.

## Reading the cost-ray image

`/cnav_cost_ray_image` is the HUD panel `SAC/play_cspace.py` draws, republished
for rviz's Image display:

```
        beam b = 0 ────────────────────────► b = 35      (10° apart)
layer k = 0  ████████████████████████████   ← robot keeps its current heading
   (θ = 0)
   │              ▓▓▓▓▓▓                      red   = obstacle close
   │              ▓▓▓▓▓▓                      green = free / far
   ▼                                          colour = (255(1-v), 255v, 40)
layer k = 29 ████████████████████████████        v = first_entry / max_range
```

- **x axis is the beam** (0..35 at 10°, so 360° across), **y axis is the θ layer**
  (0..nz-1) — row 0 is the layer where the robot holds its current heading.
  Neither axis is unwrapped or rotated; the layout is exactly play_cspace's.
- **The colour is channel 0 after normalization, not the raw distance.**
  `v = first_entry / max_range`; the image is red near an obstacle and green at
  the 3 m observation boundary. This is the single channel in the current policy.
- **A fully red image means the robot is already in collision at its current
  pose**, which is a genuine reading, not a bug: the first march step is
  `g_min/2 = 0.05 m`, so every ray clips there. Compare against the actual pose
  before concluding the perception is broken.
- **The heatmap follows the policy observation layout** used by
  `play_cspace.py`.

The image is a fixed-size raster: rviz renders it 1:1 and will not scale it, so
raise `~cost_ray_image_cell` on a large screen (10 ⇒ 360×300).

## Parameters

| param | default | meaning |
|---|---|---|
| `~model_name` | `interval_1_2000000steps` | checkpoint basename in `~model_dir` |
| `~model_dir` | `pytorch_models/interval_sweep_2m` | actor directory; relative paths resolve under `~sac_dir` |
| `~sac_dir` | `/DRL-robot-navigation/SAC` | training tree, for `cspace_cost_rays` |
| `~device` | `cuda` | `cuda` or `cpu` |
| `~stochastic` | `false` | `true` samples from the policy instead of `tanh(mean)` |
| `~g_min`, `~nz`, `~a` | `0.1`, `30`, `2π/30` | c-space cell size, θ layers, θ step |
| `~cost_ray_max_range` | `3.0` | ray horizon (m) |
| `~cost_ray_num_beams` | `36` | beams per layer |
| `~cost_ray_h_res_degrees` | `10.0` | beam spacing |
| `~cost_ray_num_intervals` | `1` | ray channels; must match the checkpoint |
| `~footprint` | `[[0.6,0.07],…]` | robot polygon, metres, +x forward |
| `~ray_frame` | `env` | `env` \| `ros` — see above |
| `~preprocess` | `interval` | divide each interval distance by max range |
| `~v_linear_max`, `~v_angular_max` | `0.5`, `2.0` | action scale (m/s, rad/s) |
| `~allow_reverse` | `true` | maps `a0` to `[0,1]` when false |
| `~velocity_filter_k` | `[0,0]` | first-order `[linear, angular]` command filter; Sparrow profiles set `[0.55,0.60]` |
| `~rate` | `10.0` | control rate (Hz); training used 0.1 s |
| `~goal_tolerance` | `0.3` | arrival radius (m); training's `target_area` |
| `~scan_max_range` | `0` | hits beyond this count as no-return; `0` ⇒ use `~cost_ray_max_range` |
| `~scan_min_range` | `0` | ignore returns closer than this |
| `~cost_ray_image_topic` | `/cnav_cost_ray_image` | cost-ray heatmap topic |
| `~cost_ray_image_cell` | `5` | pixels per cost-ray cell in that image (5 ⇒ 180×150) |
| `~publish_debug` | `true` | publish the goal marker and both cost-ray visualisations |

## Tools

- `check_models.py` — load the configured checkpoint and run one
  observation→action step; catches both a config/checkpoint geometry mismatch and
  a checkpoint too old to load (wrong kinematic block), before a rollout.
  ```bash
  rosrun cnav_ros check_models.py --config config/cnav_legacy_sparrow.yaml
  ```
  It shares `_kin_dim_of` and `KINEMATIC_DIM` with the node, so it cannot report a
  model as loadable that the node would reject.
- `send_goal.py` — publish a goal `PoseStamped`.
  ```bash
  rosrun cnav_ros send_goal.py 3.0 3.0 --frame odom
  ```

## Caveats

- **Single goal, no global planner.** The shipped checkpoints were trained with
  `settings.GLOBAL_PLAN_ENABLED = False`, i.e. against a randomly placed target
  point, so the node drives straight at one goal and stops within
  `~goal_tolerance`. Feeding it graph-planner waypoints is a shift in the target
  distribution — the policy sees distance/heading to the current waypoint, so
  it should transfer, but the waypoint statistics would not match training.
- **No collision checking or safety layer.** The actor is the only thing between
  the robot and an obstacle. Add a watchdog before running on hardware.
- The old Gazebo family is loaded but its ray pipeline is inferred, not verified
  (see above).
- `SAC/play_cspace.py:118` hardcodes `cnn_feature_dim=64`, so it cannot load the
  `SAC_sparrow_cspace_*` checkpoints (which need 128). That is a bug in that
  script, not here — this node reads the feature width from the state dict.
