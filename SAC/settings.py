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
# Continuous action convention (Sparrow c-space pipeline)
# ---------------------------------------------------------------------------
# actor 的线性输出 a0 取 [-1,1]，-1 即全速倒车，**不需要任何映射**：env 端
# _Continuous_Kinematic_model_vec 用 continuous_scale*a 直接把指令缩放成 cm/s，
# 因此 a0=-1 即 -50 cm/s，与离散动作空间的倒车档 (a_state[5]) 语义一致。
# （调用方直接把 actor 输出交给 env.step；"不允许倒车" 的前向分支已删除。）


# ---------------------------------------------------------------------------
# C-space observation geometry — 单一来源（cost-ray 观测的真实几何参数）
# 这里的 nz / 波束数 / 分辨率 同时决定:
#   * RAY_*_DIM / CNN 输入形状（下面别名）
#   * Sparrow_V2.SPARROW_CSPACE_CFG（观测管线实际读取的 cfg）
# 之前 nz/a 有两份拷贝(settings 的 RAY_V_LAYERS 与 env 的 cspace_cfg["nz"])，只改
# 一处就会维度错位(改 RAY_V_LAYERS=30 而观测仍 10 层 → reshape/load 报错)。
# 现在观测管线与网络形状都从这里派生，想调 θ 分辨率只改 CSPACE_NZ 一处。
# ---------------------------------------------------------------------------
CSPACE_COST_RAY_MAX_RANGE = 3.0     # cost-ray 最大探测范围 (m)（= 观测局部窗口半径）
CSPACE_COST_RAY_NUM_BEAMS = 36      # 水平波束数（0°–360°）
CSPACE_COST_RAY_H_RES_DEGREES = 360.0 / CSPACE_COST_RAY_NUM_BEAMS  # 10 deg
CSPACE_COST_RAY_NUM_INTERVALS = 1   # 连续径向区间数：首次命中距离、障碍宽度、空闲宽度……
CSPACE_NZ = 30                      # 观测 θ 朝向层数（cost-ray 第三维, = nz_）
CSPACE_A = 2 * math.pi / CSPACE_NZ  # θ 分辨率 (rad)，~0.209 = 12°；nz·a=2π -> nz 个
                                    # 朝向唯一、无重复拷贝（旧观测 a=0.698≈2π/9 配
                                    # nz=10 时，第 10 层是第 0 层的拷贝）

# CNN ray-grid dims (aliases kept — the network / reshape code reads these)
RAY_CHANNELS = CSPACE_COST_RAY_NUM_INTERVALS             # 每个径向区间对应一个 CNN 通道
RAY_H_BEAMS = CSPACE_COST_RAY_NUM_BEAMS                # = 36
RAY_V_LAYERS = CSPACE_NZ                               # = 30
RAY_GRID_FLAT_DIM = RAY_CHANNELS * RAY_H_BEAMS * RAY_V_LAYERS

# Kinematic block appended to the CNN cost-ray state:
# [distance_to_goal (m), heading_error (rad), normalized_linear_velocity,
#  normalized_angular_velocity].  The velocity terms are the simulator's
# post-filter *actual* velocities, not the previous action command:
# car_state[:, 3:5] / [v_linear_max, v_angular_max], clipped to [-1, 1].
# This keeps the observation Markov under Sparrow's first-order actuator model.
# The ray half of the input is the C×36×30 ray-segment grid, so checkpoints
# trained with a different channel count or the old 2-D kinematic block are
# incompatible.
KINEMATIC_DIM = 4
CNN_STATE_DIM = RAY_GRID_FLAT_DIM + KINEMATIC_DIM      # CNN state

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

# Training footprint randomization: centered, independently rotated rectangles.
# Each episode draws a new union of 1-3 rectangles per parallel environment.
RANDOM_FOOTPRINT_ENABLED = True
RANDOM_FOOTPRINT_COUNT = (1, 3)
RANDOM_FOOTPRINT_LENGTH = (0.35, 1.20)  # m
RANDOM_FOOTPRINT_WIDTH = (0.12, 0.35)   # m
RANDOM_FOOTPRINT_ANGLE = (-math.pi / 2, math.pi / 2)  # rad; 180° covers all rectangle orientations

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
LR = 1e-4             # actor & critic learning rate (pre-debug baseline)
ACTOR_LR = 5e-5       # keep separate-rate API aligned with the baseline
CRITIC_LR = 5e-5      # keep separate-rate API aligned with the baseline
ALPHA_LR = 3e-4       # temperature (entropy) learning rate (pre-debug baseline)
SAC_INITIAL_ALPHA = 1.0  # original SAC temperature initialization
EVAL_FREQ = 5e3       # steps between evaluations
EVAL_EP = 3           # number of episodes per evaluation
MAX_TIMESTEPS = 5e6   # total training steps

# ---------------------------------------------------------------------------
# Reward weighting
# ---------------------------------------------------------------------------
SAFETY_REWARD_WEIGHT = 0.0
SPEED_REWARD_WEIGHT = 0.0
SPARROW_DISTANCE_REWARD_WEIGHT = 2.0  # tuned goal-progress weight
SPARROW_FORWARD_REWARD_WEIGHT = 3.0   # tuned forward-motion reward
SPARROW_REVERSE_PENALTY_WEIGHT = 1.5  # full-speed reverse motion: -1.5 per step
SPARROW_ANGULAR_VEL_PENALTY = 0.1     # full angular command: -0.1 per step
SPARROW_STEP_PENALTY = 3.0            # tuned constant per-step penalty

# ---------------------------------------------------------------------------
# Sparrow c-space environment (Sparrow_V2 / train.py)
# ---------------------------------------------------------------------------
SPARROW_N_ENVS = 32          # vectorized environments per training step
SPARROW_N_OBSTACLES = 16      # moving obstacles per environment
SPARROW_DYNAMIC_OBS = True   # whether to spawn moving obstacles in c-space environments
SPARROW_OBS_V = 25           # maximum moving-obstacle speed (cm/s)
SPARROW_LD_NUM = 72         # lidar rays per world (ld_a_range=360 deg -> 1 deg apart)
SPARROW_LD_ACC = 8           # lidar DDA step size (cm); larger = faster but coarser scan
# 6x6 regions x 2 attempts x 0.25 -> about 18 static line obstacles per map.
SPARROW_GENERATE_RATE = 0.25

# Per-env wall: opening width = maximum footprint projection perpendicular to
# robot forward + extra. Wall angle is sampled uniformly in [0, pi).
SPARROW_CROSS_WALL = True
SPARROW_WALL_OPENINGS = 2
SPARROW_WALL_THICKNESS_CM = 16
SPARROW_WALL_GAP_EXTRA_CM = 15
SPARROW_WALL_START_OFFSET_CM = 160
SPARROW_WALL_OPENING_JITTER_CM = 18

# Stuck auto-reset: boxed-in detection via the theta=0 cost-ray layer
SPARROW_STUCK_RESET = True
SPARROW_STUCK_THRESHOLD = 0.1   # < 0 disables stuck reset (training default);
                                 # interactive play_cspace uses 0.5

# Live rendering of one parallel environment (pygame window)
SPARROW_RENDER = False           # True -> render env SPARROW_RENDER_ENV while training
SPARROW_RENDER_ENV = 0           # which env to render (0 .. SPARROW_N_ENVS-1)
SPARROW_RENDER_SPEED = 'real'    # 'slow'=30FPS / 'fast'=max / 'real'=sim-speed

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = "./pytorch_models"
RESULTS_DIR = "./results"
