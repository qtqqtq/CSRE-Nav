"""
CNN-SAC Testing Script — runs a trained CNN-SAC policy in Gazebo.

Usage:
  python3 test_cnn.py
"""

import time

import numpy as np
import torch

from laser_env import GazeboEnv
from cnn_network import CNNSAC
import settings


# ============================================================================
# Setup
# ============================================================================

file_name = "SAC_sparrow_cspace"  # name of the stored model to load

env = GazeboEnv(
    "multi_robot_scenario.launch",
    settings.ENVIRONMENT_DIM,
    use_cnn=True,
    ray_h_beams=settings.RAY_H_BEAMS,
    ray_v_layers=settings.RAY_V_LAYERS,
    max_ray_range=5.0,
)
time.sleep(5)

torch.manual_seed(settings.SEED)
np.random.seed(settings.SEED)

network = CNNSAC(
    kinematic_dim=settings.KINEMATIC_DIM,
    action_dim=settings.ACTION_DIM,
    max_action=settings.MAX_ACTION,
    h_beams=settings.RAY_H_BEAMS,
    v_layers=settings.RAY_V_LAYERS,
    cnn_feature_dim=128,
    lr=settings.LR,
    device=settings.device,
)

try:
    network.load(file_name, settings.MODEL_DIR)
    print("Loaded model: %s" % file_name)
except Exception:
    raise ValueError("Could not load the stored model parameters from "
                     "%s/%s" % (settings.MODEL_DIR, file_name))

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
    ray_grid_flat = state[:settings.RAY_GRID_FLAT_DIM]
    kinematic = state[settings.RAY_GRID_FLAT_DIM:]
    ray_grid = ray_grid_flat.reshape(1, settings.RAY_H_BEAMS, settings.RAY_V_LAYERS)

    action = network.get_action(ray_grid, kinematic, deterministic=True)

    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)
    done = 1 if episode_timesteps + 1 == settings.MAX_STEPS else int(done)

    if done:
        print("Episode %i finished, reward: %.2f" % (episode_num, reward))
        state = env.reset()
        done = False
        episode_timesteps = 0
        episode_num += 1
    else:
        state = next_state
        episode_timesteps += 1
