"""
CNN-TD3 Training Script with NavRL-style Static Obstacle Perception.

This script trains a TD3 agent where the state representation uses a
CNN feature extractor (matching NavRL's architecture) to process a 2D
ray-grid (36 horizontal beams × 4 vertical layers) generated from the
Velodyne VLP-16 point cloud.

Key differences from train_velodyne_td3.py:
  - State: ray_grid (1×36×4) + kinematic (4)  →  total 148-dim flat
  - Actor:   CNN → 128 features → [features, kinematic] → MLP → action
  - Critic:  dual CNNs → 128 features each → [features, kinematic, action] → Q
  - The CNN mirrors NavRL's 3-conv-layer architecture

Usage:
  python3 train_cnn_td3.py
"""

import os
import time

import numpy as np
import torch
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from velodyne_env import GazeboEnv
from cnn_network import CNNTD3


def evaluate(network, env, epoch, eval_episodes=10):
    """Evaluate the CNN-TD3 agent. Returns (avg_reward, avg_collision_rate)."""
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        done = False
        while not done and count < 501:
            # Unpack state: [ray_grid_flat (756), kinematic (4)]
            ray_grid_flat = state[:RAY_GRID_FLAT_DIM]
            kinematic = state[RAY_GRID_FLAT_DIM:]
            ray_grid = ray_grid_flat.reshape(1, RAY_H_BEAMS, RAY_V_LAYERS)

            action = network.get_action(ray_grid, kinematic)
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, _ = env.step(a_in)
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    print("..............................................")
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    print("..............................................")
    return avg_reward, avg_col


# ============================================================================
# Hyper-parameters (same as original TD3)
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
eval_freq = 5e3
init_step = 2e3
max_ep = 500
eval_ep = 3
max_timesteps = 5e6
expl_noise = 1
expl_decay_steps = 500000
expl_min = 0.1
batch_size = 40
discount = 0.99999
tau = 0.005
policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2
buffer_size = 250000  # 250K samples (reduced from 1M for CNN GPU memory)
file_name = "TD3_CNN_velodyne_1"
save_model = True
load_model = True
load_model_name = "TD3_CNN_velodyne"
random_near_obstacle = True

# ---- Reward weight coefficients ----
#   safety_reward_weight: scales the log-average static safety term (r_ss)
#   speed_reward_weight:  scales the velocity-direction alignment term (r_vel)
#   Higher safety weight → more conservative, obstacle-avoiding behavior
#   Higher speed weight   → more aggressive goal-seeking behavior
SAFETY_REWARD_WEIGHT = 1.0
SPEED_REWARD_WEIGHT = 1.0

# ============================================================================
# Ray-grid dimensions (from c_space_mapper config)
#   36 horizontal beams × nz_ theta orientations (nz_ = ⌈2π/a⌉ + 1, a=0.314 → 21)
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

if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

environment_dim = 40  # kept for backward compat with env (40-bin for 360° LiDAR)

env = GazeboEnv(
    "multi_robot_scenario.launch",
    environment_dim,
    use_cnn=True,
    ray_h_beams=RAY_H_BEAMS,
    ray_v_layers=RAY_V_LAYERS,
    max_ray_range=5.0,
    safety_reward_weight=SAFETY_REWARD_WEIGHT,
    speed_reward_weight=SPEED_REWARD_WEIGHT,
)
time.sleep(5)

torch.manual_seed(seed)
np.random.seed(seed)

# Create the CNN-TD3 network
network = CNNTD3(
    kinematic_dim=KINEMATIC_DIM,
    action_dim=ACTION_DIM,
    max_action=MAX_ACTION,
    h_beams=RAY_H_BEAMS,
    v_layers=RAY_V_LAYERS,
    cnn_feature_dim=128,
    lr=1e-4,
    device=device,
    safety_reward_weight=SAFETY_REWARD_WEIGHT,
    speed_reward_weight=SPEED_REWARD_WEIGHT,
)

replay_buffer = ReplayBuffer(buffer_size, seed)

if load_model:
    try:
        network.load(load_model_name, "./pytorch_models")
        print("Loaded stored model: %s" % load_model_name)
    except Exception:
        print(
            "Could not load the stored model parameters, "
            "initializing training with random parameters"
        )

# ============================================================================
# Training Loop
# ============================================================================

evaluations = []

timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1

count_rand_actions = 0
random_action = []

print("Starting CNN-TD3 training on device: %s" % device)
print("State dim: %d (ray_grid: %d + kinematic: %d)" %
      (STATE_DIM, RAY_GRID_FLAT_DIM, KINEMATIC_DIM))

while timestep < max_timesteps:

    # --- Episode termination ---
    if done:
        if timestep >= init_step:
            network.train(
                replay_buffer,
                episode_timesteps,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
            )
            torch.cuda.empty_cache()
            # Log episode-level metrics
            network.log_episode(
                episode_num=episode_num,
                reward=episode_reward,
                length=episode_timesteps,
                goal_reached=target,
                collision=(episode_reward < -90),
                expl_noise=expl_noise,
            )

        if timesteps_since_eval >= eval_freq:
            print("Validating timestep %i, episode %i" % (timestep, episode_num))
            timesteps_since_eval %= eval_freq
            avg_reward, avg_col = evaluate(
                network=network, env=env, epoch=epoch,
                eval_episodes=eval_ep)
            evaluations.append(avg_reward)
            network.log_eval(epoch, avg_reward, avg_col)
            network.save(file_name, directory="./pytorch_models")
            np.save("./results/%s" % file_name, evaluations)
            epoch += 1

        state = env.reset()
        done = False

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

        # Log cost_rays visualization at episode start
        init_rg_flat = state[:RAY_GRID_FLAT_DIM]
        cost_rays_proc = init_rg_flat.reshape(1, RAY_H_BEAMS, RAY_V_LAYERS)
        network.log_cost_rays(timestep, cost_rays_proc, env.cost_rays_data)

    # --- Exploration noise ---
    if expl_noise > expl_min:
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

    # Unpack state into ray_grid + kinematic for action inference
    ray_grid_flat = state[:RAY_GRID_FLAT_DIM]
    kinematic = state[RAY_GRID_FLAT_DIM:]
    ray_grid = ray_grid_flat.reshape(1, RAY_H_BEAMS, RAY_V_LAYERS)

    action = network.get_action(ray_grid, kinematic)
    action = (action + np.random.normal(0, expl_noise, size=action.shape)).clip(
        -MAX_ACTION, MAX_ACTION
    )

    # --- Random near-obstacle exploration ---
    if random_near_obstacle:
        if (
            np.random.uniform(0, 1) > 0.85
            and min(env.velodyne_data) < 0.6
            and count_rand_actions < 1
        ):
            count_rand_actions = np.random.randint(8, 15)
            random_action = np.random.uniform(-1, 1, 2)

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action
            action[0] = -1  # force backward

    # Step the environment
    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)
    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    episode_reward += reward

    # Store transition
    replay_buffer.add(state, action, reward, done_bool, next_state)

    # Log cost_rays periodically during episode
    if episode_timesteps % 100 == 0:
        cr_flat = state[:RAY_GRID_FLAT_DIM]
        cr_proc = cr_flat.reshape(1, RAY_H_BEAMS, RAY_V_LAYERS)
        network.log_cost_rays(timestep, cr_proc, env.cost_rays_data)

    # Update counters
    state = next_state
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# ============================================================================
# Final evaluation and save
# ============================================================================

avg_reward, avg_col = evaluate(network=network, env=env, epoch=epoch,
                                eval_episodes=eval_ep)
evaluations.append(avg_reward)
network.log_eval(epoch, avg_reward, avg_col)
if save_model:
    network.save("%s" % file_name, directory="./pytorch_models")
np.save("./results/%s" % file_name, evaluations)
print("Training complete. Model saved to ./pytorch_models/%s" % file_name)
