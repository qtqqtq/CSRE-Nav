"""
CNN-TD3 Testing Script — runs a trained CNN-TD3 policy in Gazebo.

Usage:
  python3 test_cnn_td3.py
"""

import time

import numpy as np
import torch

from velodyne_env import GazeboEnv
from cnn_network import CNNTD3


# ============================================================================
# Ray-grid dimensions (must match c_space_mapper config and training)
# ============================================================================

RAY_H_BEAMS = 36
RAY_V_LAYERS = 21    # theta orientations from c-space mapper (= nz_)
RAY_GRID_FLAT_DIM = RAY_H_BEAMS * RAY_V_LAYERS  # 756
KINEMATIC_DIM = 4
STATE_DIM = RAY_GRID_FLAT_DIM + KINEMATIC_DIM    # 760
ACTION_DIM = 2
MAX_ACTION = 1

# ============================================================================
# Setup
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
max_ep = 500

environment_dim = 40
file_name = "TD3_CNN_velodyne"  # name of the stored model to load

env = GazeboEnv(
    "multi_robot_scenario.launch",
    environment_dim,
    use_cnn=True,
    ray_h_beams=RAY_H_BEAMS,
    ray_v_layers=RAY_V_LAYERS,
    max_ray_range=5.0,
)
time.sleep(5)

torch.manual_seed(seed)
np.random.seed(seed)

network = CNNTD3(
    kinematic_dim=KINEMATIC_DIM,
    action_dim=ACTION_DIM,
    max_action=MAX_ACTION,
    h_beams=RAY_H_BEAMS,
    v_layers=RAY_V_LAYERS,
    cnn_feature_dim=128,
    lr=1e-4,
    device=device,
)

try:
    network.load(file_name, "./pytorch_models")
    print("Loaded model: %s" % file_name)
except Exception:
    raise ValueError("Could not load the stored model parameters from "
                     "./pytorch_models/%s" % file_name)

# ============================================================================
# Testing Loop
# ============================================================================

done = False
episode_timesteps = 0
episode_num = 0
state = env.reset()

print("Starting CNN-TD3 test run...")

while True:
    # Unpack state
    ray_grid_flat = state[:RAY_GRID_FLAT_DIM]
    kinematic = state[RAY_GRID_FLAT_DIM:]
    ray_grid = ray_grid_flat.reshape(1, RAY_H_BEAMS, RAY_V_LAYERS)

    action = network.get_action(ray_grid, kinematic)

    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)

    if done:
        print("Episode %i finished, reward: %.2f" % (episode_num, reward))
        state = env.reset()
        done = False
        episode_timesteps = 0
        episode_num += 1
    else:
        state = next_state
        episode_timesteps += 1
