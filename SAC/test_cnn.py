"""
CNN-SAC Testing Script — runs a trained CNN-SAC policy in Gazebo.

Usage:
  python3 test_cnn.py
"""

import time

import numpy as np
import torch

from velodyne_env import GazeboEnv
from cnn_network import CNNSAC
from settings import (
    ACTION_DIM,
    ENVIRONMENT_DIM,
    KINEMATIC_DIM,
    LR,
    MAX_ACTION,
    MAX_STEPS,
    MODEL_DIR,
    RAY_GRID_FLAT_DIM,
    RAY_H_BEAMS,
    RAY_V_LAYERS,
    SEED,
    device,
)


# ============================================================================
# Setup
# ============================================================================

file_name = "SAC_CNN_velodyne_2"  # name of the stored model to load

env = GazeboEnv(
    "multi_robot_scenario.launch",
    ENVIRONMENT_DIM,
    use_cnn=True,
    ray_h_beams=RAY_H_BEAMS,
    ray_v_layers=RAY_V_LAYERS,
    max_ray_range=5.0,
)
time.sleep(5)

torch.manual_seed(SEED)
np.random.seed(SEED)

network = CNNSAC(
    kinematic_dim=KINEMATIC_DIM,
    action_dim=ACTION_DIM,
    max_action=MAX_ACTION,
    h_beams=RAY_H_BEAMS,
    v_layers=RAY_V_LAYERS,
    cnn_feature_dim=128,
    lr=LR,
    device=device,
)

try:
    network.load(file_name, MODEL_DIR)
    print("Loaded model: %s" % file_name)
except Exception:
    raise ValueError("Could not load the stored model parameters from "
                     "%s/%s" % (MODEL_DIR, file_name))

# ============================================================================
# Testing Loop
# ============================================================================

done = False
episode_timesteps = 0
episode_num = 0
state = env.reset()

print("Starting CNN-SAC test run...")

while True:
    # Unpack state
    ray_grid_flat = state[:RAY_GRID_FLAT_DIM]
    kinematic = state[RAY_GRID_FLAT_DIM:]
    ray_grid = ray_grid_flat.reshape(1, RAY_H_BEAMS, RAY_V_LAYERS)

    action = network.get_action(ray_grid, kinematic, deterministic=True)

    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)
    done = 1 if episode_timesteps + 1 == MAX_STEPS else int(done)

    if done:
        print("Episode %i finished, reward: %.2f" % (episode_num, reward))
        state = env.reset()
        done = False
        episode_timesteps = 0
        episode_num += 1
    else:
        state = next_state
        episode_timesteps += 1
