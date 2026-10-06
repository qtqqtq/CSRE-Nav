# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DRL-robot-navigation is a **SAC (Soft Actor-Critic)** and **TD3** deep reinforcement learning system for mobile robot navigation in ROS Noetic + Gazebo. A Pioneer 3-DX robot equipped with a 2D laser scanner learns to reach random goal points while avoiding obstacles.

**Note**: TD3 has been replaced by SAC as the primary algorithm. The SAC implementation uses a stochastic policy with auto-tuned temperature α (entropy regularization). The old TD3 code can be recovered from git history (commit `ebdc8dd`).

**Paper**: "Goal-Driven Autonomous Exploration Through Deep Reinforcement Learning" (ICRA 2022 / IEEE RA-L).

## Build and Run

### Docker (headless training)

```bash
docker build -f dockerfile -t drl_noetic .
docker run -it --rm drl_noetic
# Inside container:
cd DRL-robot-navigation/SAC
python3 train_sac.py            # SAC training (MLP)
python3 train_cnn_sac.py        # SAC training (CNN)
```

### Docker with GPU + GUI (local)

```bash
bash buildContainer.bash   # requires goal_driven image
```

### Native (bare-metal ROS Noetic)

```bash
cd catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5
source devel/setup.bash

# Train
cd ../SAC
python3 train_sac.py            # SAC training (MLP)
python3 train_cnn_sac.py        # SAC training (CNN)

# Monitor training
tensorboard --logdir runs

# Test a trained model
python3 test_sac.py             # MLP SAC
python3 test_cnn.py             # CNN SAC

# Kill everything
killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient python python3
```

### Inside Docker (build script)

```bash
bash /DRL-robot-navigation/build.bash   # builds catkin_ws only
```

## Architecture

### Data flow

```
Gazebo simulation → /r1/front_laser/scan (LaserScan) → laser_callback()
  → 40-bin discretized laser distances → state vector
  → SAC Actor (stochastic) → [linear_vel, angular_vel] → /r1/cmd_vel → Gazebo
```

### State and action spaces

- **State** (44-dim MLP / 364-dim CNN): 40 laser distance bins (or 360-dim ray-grid) + `[distance_to_goal, heading_error, linear_vel, angular_vel]`
- **Action** (2-dim): `[linear_vel ∈ [-1,1] (负值即倒车), angular_vel ∈ [-1,1]]` — sent to the robot via `/r1/cmd_vel`
- **Reward**: `+100` goal reached, `-100` collision, otherwise `speed_weight * r_vel + safety_weight * r_ss` (velocity alignment + log-average safety)

### Key files

| File | Purpose |
|------|---------|
| `SAC/train_sac.py` | **SAC training (MLP)** — SACActor (400→300→action_dim×2) with stochastic policy, twin Critic, auto-tuned α |
| `SAC/train_cnn_sac.py` | **SAC training (CNN)** — CNN backbone + stochastic Actor, same SAC algorithm |
| `SAC/cnn_network.py` | CNN network definitions: `StaticObstacleCNN`, `CNNActor`, `CNNCritic`, and the `CNNSAC` agent wrapper |
| `SAC/test_sac.py` | Inference-only MLP SAC — loads trained Actor, runs deterministic eval |
| `SAC/test_cnn.py` | Inference-only CNN SAC — loads trained Actor, runs deterministic eval |
| `SAC/replay_buffer.py` | Standard deque-based experience replay buffer |
| `GDAM_env.py` | Alternative environment for the GDAM variant — 2D RPLIDAR and POI-based waypoint generation with `move_base` |

### SAC algorithm details

- **Stochastic policy**: Actor outputs Gaussian (mean, log_std); action = tanh(mean + std * ε), ε ~ N(0,I)
- **Entropy regularization**: Soft Bellman target = r + γ*(min_Q - α*log_prob)
- **Auto-tuned α**: log_α learned via gradient descent; target entropy = -action_dim
- **Twin critics** (reused from TD3): target Q = min(Q1, Q2) to reduce overestimation bias
- **No delayed updates**: Actor, critic, and α updated every training step
- **Soft target updates** with `τ=0.005` (critic target only, no actor target)
- **No external exploration noise**: SAC's stochastic policy provides inherent exploration
- **Discriminator `γ=0.95`**
- **Random near obstacle** heuristic retained for escape behavior

### Removed: colcon_ws (ROS 2)

The `colcon_ws/` directory (ROS 2 Foxy workspace with `c_space_mapper`, `c_space_mapper_msgs`, and `ros1_bridge`) has been deleted. The associated `ros2_start.bash` script was removed. The dockerfile no longer installs ROS 2. The unused ROS 1 `c_space_mapper` and `c_space_mapper_msgs` packages were also removed; current training computes cost rays in `SAC/cspace_cost_rays.py`.

## Environment variables

```
ROS_HOSTNAME=localhost
ROS_MASTER_URI=http://localhost:11311
ROS_PORT_SIM=11311
GAZEBO_RESOURCE_PATH=~/DRL-robot-navigation/catkin_ws/src/multi_robot_scenario/launch
```

## Misc

- **`.devcontainer/`** — VS Code Dev Container config pointing to `dockerfile`
- **`maps/`** — map files for the 2D navigation variant
- Trained models are saved to `SAC/pytorch_models/`, evaluation results to `SAC/results/`, TensorBoard logs to `SAC/runs/`
- 当做出较大更新时，提交git commit
