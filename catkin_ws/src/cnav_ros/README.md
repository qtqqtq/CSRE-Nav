# CNav — `cnav_ros`

ROS 1 (Noetic) wrapper for the **CNN-SAC c-space navigation policy** trained in
`DRL-robot-navigation/SAC` — the cost-ray actor from `Sparrow_V2.SparrowCspaceEnv`.

The node rebuilds the training observation from live sensor data and publishes
velocity commands:

```
/scan ────┐
/odom ────┼─→ occ grid → c-space → 36×nz cost rays → log encode ─┐
/cnav_goal┘                                                       ├─→ actor → /cnav_cmd_vel
                              [distance, heading] ─────────────────┘
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
> **The 11 checkpoints that predate this change can no longer be loaded.**
> `check_models.py` reports `1/12 ok`; the one that loads is
> **`dense_sample_cspace_3`** — retrained against the new observation — and it is
> what the node runs by default. The other eleven fail with a message naming the
> model and saying it must be retrained. That is the intended behaviour, not a
> regression — retrain with `SAC/train.py` (it reads `settings.KINEMATIC_DIM`, so
> nothing else needs editing) and the config profiles below apply to the new
> checkpoint as-is.

## Quick start

```bash
cd catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5
source devel/setup.bash

# validate every checkpoint against its config first (no ROS master needed)
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
remap. That is deliberate: `planner_bench`'s fairness contract needs each run to
pass its own frozen config, which a remap in a shared launch file cannot express.

So the table above is the API the node *declares*, and the config profile
retargets it at a concrete robot — the Limo profile maps them to `/cmd_vel`,
`/odom` and `/limo/scan`. This is the same arrangement NeuPAN has: its
`/neupan_cmd_vel` is remapped to `/cmd_vel` in its own launch file and does not
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
| `config/cnav_sparrow.yaml` | `dense_sample_cspace_{0,1,2,3}` | 36×30 | 64 | log encode, `env` frame, 0.5 m/s / 2 rad/s, 3 m, 0.1 m cells, 1.20×0.14 m footprint |
| `config/cnav_legacy_sparrow.yaml` | `SAC_sparrow_cspace{,_1..4}` | 36×10 | 128 | same, but `a=0.698` → nz=10 |
| `config/cnav_legacy_gazebo.yaml` | `SAC_cspace{,_1,_2}` | 36×10 | 128 | **experimental** — subtract encode, `ros` frame, 1.0 m/s / 1.0 rad/s, 5 m, 0.2 m cells, 0.44×0.38 m footprint |

```bash
roslaunch cnav_ros cnav.launch \
    config:=$(rospack find cnav_ros)/config/cnav_legacy_sparrow.yaml \
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

## Reading the cost-ray image

`/cnav_cost_ray_image` is the HUD panel `SAC/play_cspace.py` draws, republished
for rviz's Image display:

```
        beam b = 0 ────────────────────────► b = 35      (10° apart)
layer k = 0  ████████████████████████████   ← robot keeps its current heading
   (θ = 0)
   │              ▓▓▓▓▓▓                      red  = obstacle close
   │              ▓▓▓▓▓▓                      green = free / far
   ▼                                          colour = (255v, 255(1-v), 40)
layer k = 29 ████████████████████████████        v = clamp(log-range / max_range)
```

- **x axis is the beam** (0..35 at 10°, so 360° across), **y axis is the θ layer**
  (0..nz-1) — row 0 is the layer where the robot holds its current heading.
  Neither axis is unwrapped or rotated; the layout is exactly play_cspace's.
- **The colour is the post-`~preprocess` value, not the raw distance.** Under
  `log`, `v = clamp(ln(max_range / d) / max_range, 0, 1)` — at `max_range = 3 m`
  that is green for free space, `(93, 161, 40)` at 1 m, and saturated red near
  0.15 m. The log is the point: it magnifies near range (`|dp/dd| = 1/d`), so
  what turns red here is what the policy actually reacts to.
- **A fully red image means the robot is already in collision at its current
  pose**, which is a genuine reading, not a bug: the first march step is
  `g_min/2 = 0.05 m`, so every ray clips there. Compare against the actual pose
  before concluding the perception is broken.
- **This is deliberately not the same mapping as `c_space_mapper`'s
  `cost_ray_image`**, which is `mono8`, linear (`1 - d/max_range`), and
  horizontally unwrapped so beam 0 lands in the middle. That node's convention
  is for its own `unit/costray.rviz`; this one follows play_cspace. Two nodes,
  two topics, two mappings — not a duplicate implementation.

The image is a fixed-size raster: rviz renders it 1:1 and will not scale it, so
raise `~cost_ray_image_cell` on a large screen (10 ⇒ 360×300).

## Parameters

| param | default | meaning |
|---|---|---|
| `~model_name` | `dense_sample_cspace_3` | checkpoint basename in `~model_dir` |
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
| `~cost_ray_image_topic` | `/cnav_cost_ray_image` | cost-ray heatmap topic |
| `~cost_ray_image_cell` | `5` | pixels per cost-ray cell in that image (5 ⇒ 180×150) |
| `~publish_debug` | `true` | publish the goal marker and both cost-ray visualisations |

## Tools

- `check_models.py` — load every checkpoint against a config and run one
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
  distribution — the policy only sees distance/heading to the current waypoint, so
  it should transfer, but the waypoint statistics would not match training.
- **No collision checking or safety layer.** The actor is the only thing between
  the robot and an obstacle. Add a watchdog before running on hardware.
- The old Gazebo family is loaded but its ray pipeline is inferred, not verified
  (see above).
- `SAC/play_cspace.py:118` hardcodes `cnn_feature_dim=64`, so it cannot load the
  `SAC_sparrow_cspace_*` checkpoints (which need 128). That is a bug in that
  script, not here — this node reads the feature width from the state dict.
