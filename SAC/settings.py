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
# Continuous action mapping (Sparrow c-space pipeline)
# ---------------------------------------------------------------------------
# 是否允许倒车：为 True 时连续线速度指令 a0 取 [-1,1]（-1 = 全速倒车），
# 训练/play 的动作映射改为恒等映射；为 False 时保持原来的前向约束
# a_in[0] = (a0+1)/2 ∈ [0,1]。env 端 _Continuous_Kinematic_model_vec 用
# continuous_scale*a 直接把指令缩放成 cm/s，因此 a0=-1 即 -50 cm/s，
# 与离散动作空间的倒车档 (a_state[5]) 语义一致。
ALLOW_REVERSE = True


def map_continuous_action(action, allow_reverse=None):
    """把 actor 输出 (tanh [-1,1], shape (N,2)) 映射为 env 速度指令。

    allow_reverse=True -> 线性 [-1,1]（负值倒车）；False -> (a0+1)/2 进 [0,1]。
    角速度恒为 [-1,1]。
    """
    if allow_reverse is None:
        allow_reverse = ALLOW_REVERSE
    if allow_reverse:
        return action
    a0, a1 = action[..., 0], action[..., 1]
    return torch.stack([(a0 + 1) / 2, a1], dim=-1)

# ---------------------------------------------------------------------------
# C-space observation geometry — 单一来源（cost-ray 观测的真实几何参数）
# 这里的 nz / 波束数 / 分辨率 同时决定:
#   * RAY_*_DIM / CNN 输入形状（下面别名）
#   * Sparrow_V2.SPARROW_CSPACE_CFG（观测管线实际读取的 cfg）
# 之前 nz/a 有两份拷贝(settings 的 RAY_V_LAYERS 与 env 的 cspace_cfg["nz"])，只改
# 一处就会维度错位(改 RAY_V_LAYERS=30 而观测仍 10 层 → reshape/load 报错)。
# 现在观测管线与网络形状都从这里派生，想调 θ 分辨率只改 CSPACE_NZ 一处。
# 注意: cspace_cost_rays.DEFAULT_CFG 仍是 Gazebo/C++ 参考配置(0.44 m 车 / 5 m 量程，
# a=0.698/nz=10)，仅供该模块 --bench / verify 使用；实际训练评测走下面的值。
# ---------------------------------------------------------------------------
CSPACE_G_MIN = 0.1                  # c-space 网格分辨率 (m)
CSPACE_COST_RAY_MAX_RANGE = 3.0     # cost-ray 最大探测范围 (m)（= 观测局部窗口半径）
CSPACE_COST_RAY_NUM_BEAMS = 36      # 水平波束数（0°–360°）
CSPACE_COST_RAY_H_RES_DEGREES = 360.0 / CSPACE_COST_RAY_NUM_BEAMS  # 10 deg
CSPACE_NZ = 30                      # 观测 θ 朝向层数（cost-ray 第三维, = nz_）
CSPACE_A = 2 * math.pi / CSPACE_NZ  # θ 分辨率 (rad)，~0.209 = 12°；nz·a=2π -> nz 个
                                    # 朝向唯一、无重复拷贝（旧观测 a=0.698≈2π/9 配
                                    # nz=10 时，第 10 层是第 0 层的拷贝；见 PLANNER_A 注释）

# CNN ray-grid dims (aliases kept — the network / reshape code reads these)
RAY_H_BEAMS = CSPACE_COST_RAY_NUM_BEAMS                # = 36
RAY_V_LAYERS = CSPACE_NZ                               # = 30
RAY_GRID_FLAT_DIM = RAY_H_BEAMS * RAY_V_LAYERS         # 1080 (36 beams × 30 layers)

# Kinematic block appended to the CNN cost-ray state: [distance_to_goal (m),
# heading_error (rad)].  故意不再等于 ROBOT_DIM —— 2D 激光 MLP 那条支线的 state
# 仍然是 [.., linear_vel, angular_vel]，两者维度从此独立。
# 这里原本还追加了上一帧的动作指令 (v_lin, v_ang) = _action_to_v(current_a)，
# 已去掉：观测不再包含动作反馈。旧 checkpoint 的 actor 第一层是 feat+4，
# 因此改前训练的模型都不能再加载，需要重新训练。
KINEMATIC_DIM = 2
CNN_STATE_DIM = RAY_GRID_FLAT_DIM + KINEMATIC_DIM      # 1082 (CNN state)

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
DISCOUNT = 0.995       # gamma — discounted future reward
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
SPARROW_DISTANCE_REWARD_WEIGHT = 2.0  # R_distance: normalized progress toward the target

# ---------------------------------------------------------------------------
# Sparrow c-space environment (Sparrow_V2 / train.py)
# ---------------------------------------------------------------------------
SPARROW_N_ENVS = 32          # vectorized environments per training step
SPARROW_N_OBSTACLES = 8      # moving obstacles per environment
SPARROW_DYNAMIC_OBS = True   # whether to spawn moving obstacles in c-space environments
SPARROW_OBS_V = 25           # maximum moving-obstacle speed (cm/s)
SPARROW_LD_ACC = 8           # lidar DDA step size (cm); larger = faster but coarser scan
# 6x6 regions x 2 attempts x 0.25 -> about 18 static line obstacles per map.
SPARROW_GENERATE_RATE = 0.25

# Stuck auto-reset: boxed-in detection via the theta=0 cost-ray layer
SPARROW_STUCK_RESET = True
SPARROW_STUCK_THRESHOLD = 0.1   # < 0 disables stuck reset (training default);
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
PLANNER_FOOTPRINT = ROBOT_FOOTPRINT
PLANNER_G_MIN = 0.1              # c-space 空间分辨率 (m)，与观测管线一致
# 全局规划器自己的 θ 采样（可独立于局部观测层 settings.CSPACE_A=2π/CSPACE_NZ）。
# a 由 nz 推出保证自洽（nz·a = 2π）：nz=30 → a≈0.209 rad（12°），30 个唯一朝向，
# 不会出现旧观测 a=0.698(≈2π/9)&nz=10 时那种 θ 层重复（9 层 = 360° 后只剩拷贝）。
PLANNER_NZ = 30                  # 全局规划器的 θ 层数（与观测层 CSPACE_NZ 同取 30，可独立调整）
PLANNER_A = 2 * math.pi / PLANNER_NZ   # 全局 θ 分辨率 (rad)，由 PLANNER_NZ 推出
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
