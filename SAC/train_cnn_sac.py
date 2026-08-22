"""
CNN-SAC Training Script with NavRL-style Static Obstacle Perception.

This script trains a SAC agent where the state representation uses a
CNN feature extractor (matching NavRL's architecture) to process a 2D
ray-grid (36 horizontal beams × 10 vertical layers) generated from a
2D laser scan via the c-space mapper.

SAC uses a stochastic policy with entropy regularization and auto-tuned
temperature alpha, replacing the deterministic TD3 policy.

Usage:
  python3 train_cnn_sac.py
"""

import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from laser_env import GazeboEnv
from cnn_network import CNNSAC
import settings


def evaluate(network, env, epoch, eval_episodes=10):
    """Evaluate the CNN-SAC agent. Returns (avg_reward, avg_collision_rate)."""
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        done = False
        while not done and count < 501:
            # Unpack state: [ray_grid_flat (756), kinematic (4)]
            ray_grid_flat = state[:settings.RAY_GRID_FLAT_DIM]
            kinematic = state[settings.RAY_GRID_FLAT_DIM:]
            ray_grid = ray_grid_flat.reshape(1, settings.RAY_H_BEAMS, settings.RAY_V_LAYERS)

            action = network.get_action(ray_grid, kinematic, deterministic=True)
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
# Hyper-parameters (SAC)
# ============================================================================

# Script-specific parameters (shared values come from settings.py)
batch_size = 256
buffer_size = 250000  # 250K samples (reduced from 1M for CNN GPU memory)
file_name = "SAC_cspace_2"
save_model = True
load_model = True  # SAC has different architecture from TD3
load_model_name = "SAC_cspace_1"
random_near_obstacle = False

# ---- Reward weight coefficients ----
#   settings.SAFETY_REWARD_WEIGHT: scales the log-average static safety term (r_ss)
#   settings.SPEED_REWARD_WEIGHT:  scales the velocity-direction alignment term (r_vel)
#   Higher safety weight → more conservative, obstacle-avoiding behavior
#   Higher speed weight   → more aggressive goal-seeking behavior
# (reward weights and ray-grid dimensions are defined in settings.py)

# ============================================================================
# Setup
# ============================================================================

if not os.path.exists(settings.RESULTS_DIR):
    os.makedirs(settings.RESULTS_DIR)
if save_model and not os.path.exists(settings.MODEL_DIR):
    os.makedirs(settings.MODEL_DIR)

env = GazeboEnv(
    "multi_robot_scenario.launch",
    settings.ENVIRONMENT_DIM,
    use_cnn=True,
    ray_h_beams=settings.RAY_H_BEAMS,
    ray_v_layers=settings.RAY_V_LAYERS,
    max_ray_range=5.0,
    safety_reward_weight=settings.SAFETY_REWARD_WEIGHT,
    speed_reward_weight=settings.SPEED_REWARD_WEIGHT,
)
time.sleep(5)

torch.manual_seed(settings.SEED)
np.random.seed(settings.SEED)

# Create the CNN-SAC network
network = CNNSAC(
    kinematic_dim=settings.KINEMATIC_DIM,
    action_dim=settings.ACTION_DIM,
    max_action=settings.MAX_ACTION,
    h_beams=settings.RAY_H_BEAMS,
    v_layers=settings.RAY_V_LAYERS,
    cnn_feature_dim=128,
    lr=settings.LR,
    alpha_lr=settings.ALPHA_LR,
    device=settings.device,
    safety_reward_weight=settings.SAFETY_REWARD_WEIGHT,
    speed_reward_weight=settings.SPEED_REWARD_WEIGHT,
)

replay_buffer = ReplayBuffer(buffer_size, settings.SEED)

if load_model:
    try:
        network.load(load_model_name, settings.MODEL_DIR)
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

print("Starting CNN-SAC training on device: %s" % settings.device)
print("State dim: %d (ray_grid: %d + kinematic: %d)" %
      (settings.CNN_STATE_DIM, settings.RAY_GRID_FLAT_DIM, settings.KINEMATIC_DIM))

while timestep < settings.MAX_TIMESTEPS:

    # --- Episode termination ---
    if done:
        if timestep >= settings.WARMUP_STEPS:
            network.train(
                replay_buffer,
                episode_timesteps,
                batch_size,
                settings.DISCOUNT,
                settings.TAU,
            )
            torch.cuda.empty_cache()
            # Log episode-level metrics
            network.log_episode(
                episode_num=episode_num,
                reward=episode_reward,
                length=episode_timesteps,
                goal_reached=target,
                collision=(episode_reward < -90),
                expl_noise=0.0,
            )

        if timesteps_since_eval >= settings.EVAL_FREQ:
            print("Validating timestep %i, episode %i" % (timestep, episode_num))
            timesteps_since_eval %= settings.EVAL_FREQ
            avg_reward, avg_col = evaluate(
                network=network, env=env, epoch=epoch,
                eval_episodes=settings.EVAL_EP)
            evaluations.append(avg_reward)
            network.log_eval(epoch, avg_reward, avg_col)
            network.save(file_name, directory=settings.MODEL_DIR)
            np.save("%s/%s" % (settings.RESULTS_DIR, file_name), evaluations)
            epoch += 1

        state = env.reset()
        done = False

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

        # Log cost_rays visualization at episode start
        init_rg_flat = state[:settings.RAY_GRID_FLAT_DIM]
        cost_rays_proc = init_rg_flat.reshape(1, settings.RAY_H_BEAMS, settings.RAY_V_LAYERS)
        network.log_cost_rays(timestep, cost_rays_proc, env.cost_rays_data)

    # Unpack state into ray_grid + kinematic for action inference
    ray_grid_flat = state[:settings.RAY_GRID_FLAT_DIM]
    kinematic = state[settings.RAY_GRID_FLAT_DIM:]
    ray_grid = ray_grid_flat.reshape(1, settings.RAY_H_BEAMS, settings.RAY_V_LAYERS)

    action = network.get_action(ray_grid, kinematic)

    # --- Random near-obstacle exploration ---
    if random_near_obstacle:
        if (
            np.random.uniform(0, 1) > 0.85
            and min(env.laser_data) < 0.6
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
    done_bool = 0 if episode_timesteps + 1 == settings.MAX_STEPS else int(done)
    done = 1 if episode_timesteps + 1 == settings.MAX_STEPS else int(done)
    episode_reward += reward

    # Store transition
    replay_buffer.add(state, action, reward, done_bool, next_state)

    # Log cost_rays periodically during episode
    if episode_timesteps % 100 == 0:
        cr_flat = state[:settings.RAY_GRID_FLAT_DIM]
        cr_proc = cr_flat.reshape(1, settings.RAY_H_BEAMS, settings.RAY_V_LAYERS)
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
                                eval_episodes=settings.EVAL_EP)
evaluations.append(avg_reward)
network.log_eval(epoch, avg_reward, avg_col)
if save_model:
    network.save("%s" % file_name, directory=settings.MODEL_DIR)
np.save("%s/%s" % (settings.RESULTS_DIR, file_name), evaluations)
print("Training complete. Model saved to %s/%s" % (settings.MODEL_DIR, file_name))
