# multi_robot_scenario

Gazebo worlds and robot models. This package builds nothing — it is data that
another package's launch file consumes.

## Bringing up the robot system, in two launches

The environment and the controller are separate launches, so the same world
serves any planner:

```bash
# step 1: Gazebo, the world, the Limo, its controllers.  No planner.
roslaunch multi_robot_scenario multi_robot_scenario.launch

# step 2: a controller
roslaunch cnav_ros cnav.launch
```

A bare arena instead of the obstacle course, if you want to drive around
empty space:

```bash
roslaunch multi_robot_scenario multi_robot_scenario.launch \
    world_name:=$(find multi_robot_scenario)/worlds/empty.world obstacles:=false
```

Spawn pose and namespace are arguments: `x`, `y`, `z`, `yaw`,
`robot_namespace`. `gui:=false` is headless.

### What comes up

| topic | from |
|---|---|
| `/limo/scan` | `gazebo_ros_laser`, 720 rays over 240°, `laser_link`, 0.05–8 m |
| `/odom` | `gazebo_ros_skid_steer_drive`, `odom` → `base_footprint` |
| `/cmd_vel` | the same plugin |
| `/limo/imu`, `/limo/depth/image_raw` | IMU and depth camera |
| `/obstacles_velocity` | `rvo_node`, when `obstacles:=true` |

`/limo/scan` is **absolute and hardcoded** in the laser plugin, so it keeps that
name even though `/odom` and `/cmd_vel` are relative to `robot_namespace` and so
land at the root only under the default `robot_namespace:=/`. The model is named
`limo`.

## `worlds/limo_complex.world`

A 20-obstacle arena, and the obstacles move. It is the environment NeuPAN ships
as `neupan_ros/example/gazebo_limo`, vendored here so the scenario does not
change when that repository does.

Ten of them are the non-convex `cross` shape, spawned by a Gazebo
`<population>`; the other ten are cylinders. Each carries a `libmulti_obs`
plugin from `rvo_ros`, which is a pure actuator: it subscribes to
`/obstacles_velocity` and calls `SetLinearVel`. `rvo_node` computes those
velocities with RVO2 from `/gazebo/model_states`, and `set_goals_client` gives
the agents the rectangle to wander in. All three are needed — drop
`set_goals_client` and every agent's preferred velocity is zero, RVO holds them
still, and the arena looks static.

Two changes from upstream's copy:

- **Physics is 0.01 / 100, not 0.002 / 1000.** The Limo is spawned into this
  world, and sub-10 ms steps blow the ODE solver up on its ground contacts —
  gzserver dies in `libgazebo_ode` right after reporting a successful spawn,
  which reads as "the planner never moves the robot". Every other world this
  repo runs is 0.01/100. Measured stable here: a 6 s drive at 0.4 m/s moved the
  robot 2.37 m with no NaN poses.
- **The `rescue randy` meshes are local.** Upstream references them by
  `https://fuel.ignitionrobotics.org/...`, and the URL contains a literal space.
  They are now `models/rescue_randy`, reached through `GAZEBO_MODEL_PATH`, which
  the launch file sets to this package's `models/` plus
  `/usr/share/gazebo-11/models`.

> **On upstream's `rvo_node` segfault.** `planner_bench/launch/sim.launch` and
> its README record that `rvo_node` exits `-11`, which is why that package runs a
> static arena instead. That does not reproduce here: `rvo_node` ran 60 s at
> state `Rsl` with a live publisher on `/obstacles_velocity`, and the obstacles
> moved 4.5–5.2 m in 20 s. The two observations are not reconciled — if you see
> the crash, it is real and this note is the thing that is wrong.

### Loading the world directly with Gazebo Sim

From this package's `worlds/` directory, add the local models to Gazebo Sim's
resource search path before loading the world:

```bash
export GZ_SIM_RESOURCE_PATH="$(realpath ../models)${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
gz sim limo_complex.world
```

`GAZEBO_MODEL_PATH`, set by the ROS launch above, is for Gazebo Classic.
Loading the world in Gazebo Sim does not port the Classic `libmulti_obs.so`
plugin or start the ROS obstacle controller and Limo spawning workflow. Use
the ROS launch above in the ROS Noetic / Gazebo Classic environment for that
complete system.

## Vendored robot

`xacro/limo/`, `meshes/limo/` and `config/limo_four_diff_control.yaml` are copies
of `limo_description` and `limo_gazebo_sim`, which live in the `limo_ros`
checkout. Behaviour is verbatim; only the paths moved:

- `$(find limo_description)/urdf/...` → `$(find multi_robot_scenario)/xacro/limo/...`
- `package://limo_description/meshes/limo_*.dae` → `package://multi_robot_scenario/meshes/limo/...`
- `limo_four_diff.xacro` gained `<xacro:arg name="robot_namespace" default="/"/>`.
  It always *consumed* that arg, in the included `limo_four_diff.gazebo`, but
  never declared it, so every caller had to pass it or xacro aborted with
  "undefined arg".

Only the two `.dae` meshes the URDF actually references were copied. The two
`.stl` files beside them (44 MB) are referenced by nothing in the upstream tree.

`limo_ros` is left untouched — it is a nested checkout, so these are copies, not
moves, and its working tree stays clean.

## The Pioneer is still here

`pioneer3dx.gazebo.launch`, `xacro/p3dx/` and `meshes/p3dx/` are all still on
disk and still used. `SAC/assets/multi_robot_scenario.launch` includes
`pioneer3dx.gazebo.launch`, so the MLP/CNN training environments in `SAC/` are
unaffected by the Limo becoming what *this* package's `multi_robot_scenario.launch`
spawns.
