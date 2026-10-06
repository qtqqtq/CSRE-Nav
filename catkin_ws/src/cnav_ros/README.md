# CNav — `cnav_ros`

ROS 1 (Noetic) wrapper for the **CNN-SAC c-space navigation policy** trained in
`DRL-robot-navigation/SAC` — the cost-ray actor from `Sparrow_V2.SparrowCspaceEnv`.

The node rebuilds the training observation from live sensor data and publishes
velocity commands:

```
/scan ────┐
/odom ────┼─→ c-space geometry → 1×36×nz first-entry rays ────┐
/cnav_goal┘                                                   ├─→ actor → velocity filter → /cnav_cmd_vel
             [distance, heading, actual v, actual w] ─────────┘
```

**Self-contained.** The checkpoints ship in `model/` and the c-space geometry
lives in `src/cnav_cspace.py` — a vendored, byte-identical copy of the training
tree's `SAC/cspace_cost_rays.py` (both backends). Nothing here reads a SAC
checkout, and no module is imported across the two trees. The node needs only a
laser scan, odometry and a tf tree.

Two geometries produce that ray grid, selected by `~backend` (see
[The c-space backend](#the-c-space-backend)):

- **`points`** (the shipped profile) — the obstacles *are* the lidar hit points
  and the footprint is intersected with them in closed form. No grid, no
  rasterisation, no march.
- **`grid`** — the older path: a local occupancy grid keeping observed-free cells
  along beam paths, occupied cells at finite hit endpoints and unknown cells
  outside observed paths, then a rasterised c-space that the rays march through.
  In configuration space, obstacle overlap is a collision; an otherwise
  obstacle-free pose is invalid only when its entire footprint is unknown. A
  footprint with any known-free overlap remains valid even if part of it is
  unknown.

Both emit the same shapes with the same semantics; they differ in geometry.

> ### Current model: one ray interval plus measured velocity
>
> `model/exact_castray_interval1_0_actor.pth` — the state is
> **1084-dimensional**: one `36×30` ray-distance channel plus
> `[distance, heading, normalized_actual_v, normalized_actual_w]`. Velocity comes
> from the filtered simulator state during training and odometry during deployment;
> it is not the previous action command.
>
> It was trained with the **point-set** backend, so `config/cnav_sparrow.yaml`
> sets `backend: points`. Older checkpoints are intentionally incompatible and
> must be retrained.

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

`~model_dir` defaults to `model`, i.e. the `model/` directory **inside this
package**; that is where `exact_castray_interval1_0_actor.pth` lives, so a fresh
checkout runs as-is. `~device` defaults to `cuda` in the configs — `cpu` is
plenty for a single 36×nz grid.

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

The SAC training tree holds several checkpoint families that are **not**
interchangeable. The network architecture is sniffed from the state dict
automatically, but the **geometry, the backend and the ray pipeline are not
recoverable from a checkpoint**, so they come from a config file. The node
validates the geometry against the loaded network and **aborts on a mismatch**
rather than navigating with a wrong observation (which would be silent and
dangerous).

| config | use with | ray grid | features | pipeline |
|---|---|---|---|---|
| `config/cnav_sparrow.yaml` | `model/exact_castray_interval1_0_actor.pth` | 1×36×30 | 64 | point-set backend, normalized first-entry distance, measured velocity, `env` frame, 0.5 m/s / 2 rad/s, 3 m, 0.1 m cells, 1.20×0.14 m footprint |
| `config/cnav_legacy_sparrow.yaml` | old one-channel checkpoints | incompatible | — | retained only as a historical geometry reference; retraining is required, and its checkpoint is not shipped |

To deploy a different checkpoint, drop its `_actor.pth` into `model/` and point
`~model_name` / `~model_dir` at it. It must come from a training run whose
geometry matches the profile — in particular, a grid-trained actor has to be
deployed with `backend: grid`, or it silently sees the wrong observation:

```bash
roslaunch cnav_ros cnav.launch \
    config:=$(rospack find cnav_ros)/config/cnav_legacy_sparrow.yaml \
    model_name:=SAC_sparrow_cspace_2
```

> The old one-channel checkpoints used a different observation shape and need
> retraining for the current pipeline.

## The c-space backend

`~backend` picks the geometry that turns the scan into cost rays. Both fill the
same `1×36×nz` channel with the same convention (distance until the footprint
first touches an obstacle, clipped to `max_range`), and both use the same
footprint, `nz`, `a` and beam spacing — so the network cannot tell them apart by
shape. What differs is the obstacle model:

| | `points` | `grid` |
|---|---|---|
| obstacle | the lidar hit points themselves | occupied cells of a 65×65 grid |
| geometry | analytic half-plane intersection against the footprint edges | footprint rasterised into a c-space, rays marched through it |
| cell size | none — metric | `g_min` = 0.1 m |
| dilation | none | `g_min` added around every hit |

The point-set route drops two approximations the grid route cannot: the 0.1 m
quantisation of obstacle positions, and the `+g_min` relaxation that makes a gap
look wider than it is. With 36 beams 10° apart, hit spacing grows with range
(28 cm at 1.6 m, 52 cm at 3 m), so the grid path can open a phantom corridor
where the robot does not actually fit — the gap between two hit endpoints
overestimates it by up to `g_min`. The point-set geometry removes the `g_min`
term; the remaining phantom gap is hit spacing alone.

**A checkpoint must be deployed on the backend it was trained with.** Switching
backends changes the world the policy sees; it is not a rendering option. When
comparing the two, use the same weights and the same seed, so the only difference
is the geometry.

`~backend` is validated at startup and the node refuses anything but `grid` or
`points`.

## Two details that are easy to get wrong

These are load-bearing; getting either wrong yields a plausible-looking but wrong
policy, which is why they are explicit config knobs rather than hardcoded.

1. **Grid handedness.** `SparrowCspaceEnv._build_occ_grid` projects hits with
   `rel_y = dx·sin(t) + dy·cos(t)`, so the c-space frame's second axis runs to the
   robot's **right** — the mirror of the standard ROS body frame. Feed ROS-frame
   hits directly and every ray comes out left/right swapped (differences up to the
   full 3 m range). `ray_frame: env` negates the ROS left component before it
   reaches the geometry; `ray_frame: ros` is the un-mirrored variant the old C++
   mapper produced. This applies to **both** backends — `points` negates the hit
   points, `grid` negates before quantisation.

2. **Velocity means measured motion, not the previous command.** Training divides
   Sparrow's post-filter `car_state` velocity by `(v_linear_max, v_angular_max)`;
   deployment applies the same normalization to odometry twist. Both values are
   clipped to `[-1,1]`. Checkpoints without this complete layout are rejected.

Also worth knowing: the grid backend quantises the scan to 8 cm, marks hit
endpoints occupied and samples beam paths as known-free, then marches rays in
0.05 m steps; the point-set backend keeps the hits at full metric precision and
solves for the contact distance directly. A real laser gives more precise hits
than the sim's 8 cm quantisation — that is the intended sim→real direction, not
a bug, and the point-set backend is what lets it through.

## Reading the cost-ray image

`/cnav_cost_ray_image` is the observation the actor consumes, drawn as a HUD
panel (`SAC/play_cspace.py`'s layout) and republished for rviz's Image display:

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
- **A fully red image is usually a phantom obstacle next to the robot, not a
  collision.** Every ray reads `entry = 0` when a hit point sits inside the
  footprint. The usual cause is a laser self-return: any robot geometry or dust
  within a few centimetres feeds a return that lands under the footprint and
  reddens all 30 θ layers at once. Raise `~scan_min_range` above it — the shipped
  profile uses `0.09`. Genuine collision reads the same way, so check the pose
  before blaming the perception.
- **The heatmap follows the policy observation layout** used by
  `play_cspace.py`.

The image is a fixed-size raster: rviz renders it 1:1 and will not scale it, so
raise `~cost_ray_image_cell` on a large screen (10 ⇒ 360×300).

## Parameters

| param | default | meaning |
|---|---|---|
| `~model_name` | `exact_castray_interval1_0` | checkpoint basename in `~model_dir`; `_actor.pth` is appended |
| `~model_dir` | `model` | actor directory; relative paths resolve inside the `cnav_ros` package |
| `~device` | `cuda` | `cuda` or `cpu` |
| `~stochastic` | `false` | `true` samples from the policy instead of `tanh(mean)` |
| `~backend` | `grid` | `grid` \| `points` — the c-space geometry; must match the checkpoint |
| `~g_min`, `~nz`, `~a` | `0.1`, `30`, `2π/30` | c-space cell size (`grid` only), θ layers, θ step |
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
  It shares `_kin_dim_of`, `KINEMATIC_DIM` and the backend's geometry with the
  node, so it cannot report a model as loadable that the node would reject.
- `send_goal.py` — publish a goal `PoseStamped`.
  ```bash
  rosrun cnav_ros send_goal.py 3.0 3.0 --frame odom
  ```

## Caveats

- **Single goal.** The shipped checkpoints were trained against a randomly
  placed target point. The node drives toward one goal and stops within
  `~goal_tolerance`.
- **No collision checking or safety layer.** The actor is the only thing between
  the robot and an obstacle. Add a watchdog before running on hardware.
- **Only the point-set backend is exercised.** The shipped profile and the smoke
  test both use `backend: points`. `grid` still runs and produces the same
  observation shapes, but nothing here is currently deployed on it.
- The old Gazebo family is loaded but its ray pipeline is inferred, not verified
  (see above).
- `SAC/play_cspace.py:118` hardcodes `cnn_feature_dim=64`, so it cannot load the
  `SAC_sparrow_cspace_*` checkpoints (which need 128). That is a bug in that
  script, not here — this node reads the feature width from the state dict.
- `src/cnav_cspace.py` is a vendored copy of the training helpers. It is
  byte-identical today, but it no longer tracks `SAC/cspace_cost_rays.py`: if the
  training-side geometry changes, this copy has to be re-synced by hand.
