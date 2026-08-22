"""
Shared configuration for the SAC robot-navigation training / test scripts.

Centralizes constants that were previously duplicated across the training,
testing, and reward-evaluation scripts so they can be changed in one place.

Usage:
    from settings import *
    # or
    from settings import ENVIRONMENT_DIM, ACTION_DIM, MODEL_DIR, ...
"""

import math

import torch

# ---------------------------------------------------------------------------
# Device / random seed
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
SEED = 0  # random seed number

# ---------------------------------------------------------------------------
# State & action dimensions (MLP / 2D laser)
# ---------------------------------------------------------------------------
ENVIRONMENT_DIM = 40  # number of discretized laser distance bins (360° 2D laser)
ROBOT_DIM = 4         # [distance_to_goal, heading_error, linear_vel, angular_vel]
STATE_DIM = ENVIRONMENT_DIM + ROBOT_DIM  # 44 (MLP state)
ACTION_DIM = 2        # [linear_vel, angular_vel]
MAX_ACTION = 1        # actor output is tanh-bounded to [-1, 1]

# ---------------------------------------------------------------------------
# CNN ray-grid dimensions (StaticObstacleCNN / NavRL-style state)
# ---------------------------------------------------------------------------
RAY_H_BEAMS = 36      # horizontal beams (0°–360°)
RAY_V_LAYERS = 10     # theta orientations from c-space mapper (= nz_)
RAY_GRID_FLAT_DIM = RAY_H_BEAMS * RAY_V_LAYERS  # 360 (36 beams × 10 layers)
KINEMATIC_DIM = ROBOT_DIM  # same kinematic state vector as the MLP variant
CNN_STATE_DIM = RAY_GRID_FLAT_DIM + KINEMATIC_DIM  # 364 (CNN state)

# ---------------------------------------------------------------------------
# Robot footprint (c-space dilation polygon, metres)
# ---------------------------------------------------------------------------
def circular_footprint(radius, n=16):
    """Regular n-gon approximating a circle of `radius` (metres)."""
    return [
        [radius * math.cos(2 * math.pi * i / n),
         radius * math.sin(2 * math.pi * i / n)]
        for i in range(n)
    ]

ROBOT_FOOTPRINT_RADIUS = 0.14   # m (= Sparrow collision_trsd = car_radius 9cm + 5cm)
ROBOT_FOOTPRINT = circular_footprint(ROBOT_FOOTPRINT_RADIUS, n=16)

# ---------------------------------------------------------------------------
# SAC policy bounds
# ---------------------------------------------------------------------------
LOG_STD_MIN = -20
LOG_STD_MAX = 2

# ---------------------------------------------------------------------------
# Episode / training-loop limits
# ---------------------------------------------------------------------------
MAX_STEPS = 500       # maximum number of steps per episode (a.k.a. max_ep)
WARMUP_STEPS = 5e3    # collect this many steps (fill replay buffer) before training begins

# ---------------------------------------------------------------------------
# SAC hyper-parameters (shared defaults; batch/buffer size are per-script)
# ---------------------------------------------------------------------------
DISCOUNT = 0.95       # gamma — discounted future reward
TAU = 0.005           # soft target update rate
LR = 1e-4             # actor & critic learning rate
ALPHA_LR = 3e-4       # temperature (entropy) learning rate
EVAL_FREQ = 5e3       # steps between evaluations
EVAL_EP = 3           # number of episodes per evaluation
MAX_TIMESTEPS = 5e6   # total training steps

# ---------------------------------------------------------------------------
# Reward weighting
# ---------------------------------------------------------------------------
SAFETY_REWARD_WEIGHT = 0.0
SPEED_REWARD_WEIGHT = 0.0

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = "./pytorch_models"
RESULTS_DIR = "./results"
