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
def rectangular_footprint(length, width):
    """Axis-aligned rectangle footprint (long axis along +x = forward), metres."""
    hl, hw = length / 2.0, width / 2.0
    return [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]]

ROBOT_FOOTPRINT_LENGTH = 1.20   # m, 沿 +x(前进)
ROBOT_FOOTPRINT_WIDTH = 0.14    # m, 沿 +y(横向)
ROBOT_FOOTPRINT = rectangular_footprint(ROBOT_FOOTPRINT_LENGTH, ROBOT_FOOTPRINT_WIDTH)

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
# Sparrow c-space environment (sparrow_cspace_env / train_sparrow_cnn_sac)
# ---------------------------------------------------------------------------
SPARROW_N_ENVS = 32          # vectorized environments per training step
SPARROW_N_OBSTACLES = 15     # obstacles per environment
SPARROW_DYNAMIC_OBS = False  # whether to spawn moving obstacles (c-space env: off)
SPARROW_LD_ACC = 8           # lidar DDA step size (cm); larger = faster but coarser scan

# Stuck auto-reset: boxed-in detection via the theta=0 cost-ray layer
SPARROW_STUCK_RESET = True
SPARROW_STUCK_THRESHOLD = 0.5   # < 0 disables stuck reset (training default);
                                 # interactive play_cspace uses 0.5

# Live rendering of one parallel environment (pygame window)
SPARROW_RENDER = False           # True -> render env SPARROW_RENDER_ENV while training
SPARROW_RENDER_ENV = 0           # which env to render (0 .. SPARROW_N_ENVS-1)
SPARROW_RENDER_SPEED = 'real'    # 'slow'=30FPS / 'fast'=max / 'real'=sim-speed

# ---------------------------------------------------------------------------
# Global planner (GraphPlanner) -- 单机器人，位形空间中的图上规划
# ---------------------------------------------------------------------------
# footprint 用 28x28 cm 方（≈碰撞圆 collision_trsd=14cm），而非观测用的 1.2m 大
# 足迹（否则会挡住 > 一半地图）。见 global_planner.py 模块 docstring。
GLOBAL_PLAN_ENABLED = False      # 训练时是否启用全局路径航点引导
PLANNER_FOOTPRINT = [[0.14, 0.14], [0.14, -0.14], [-0.14, -0.14], [-0.14, 0.14]]
PLANNER_G_MIN = 0.1              # c-space 空间分辨率 (m)，与观测管线一致
PLANNER_NZ = 10                  # θ 层数，与观测管线一致
PLANNER_STRIDE_IJ = 2            # 顶点 (i,j) 采样步长 (格)
PLANNER_STRIDE_K = 1             # 顶点 θ 层采样步长 (层)
PLANNER_CONNECT_DIST_CM = 60     # 顶点连接半径 (cm)
PLANNER_WP_REACH_CM = 60         # 距当前航点多近则前进到下一航点 (> target_area 防提前获胜)
PLANNER_MIN_WP_SPACING_CM = 40   # 航点最小间距 (> target_area 防提前获胜)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = "./pytorch_models"
RESULTS_DIR = "./results"
