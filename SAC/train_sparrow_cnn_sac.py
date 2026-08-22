"""
Vectorized CNN-SAC training on Sparrow-V2 with a c-space (NavRL) observation.

Trains the existing `cnn_network.CNNSAC` on `sparrow_cspace_env.SparrowCspaceEnv`
instead of the Gazebo `laser_env.GazeboEnv`, collecting N transitions per step
(one per vectorized env) so both the simulator and the actor forward pass run
batched on the GPU.

State / action conventions are identical to `train_cnn_sac.py`:
    - state  : (N, 364) = flatten(preprocessed cost_rays) + [dist, heading, v_lin, v_ang]
    - action : actor outputs tanh(-1,1); env receives a_in = [(a0+1)/2, a1]
               (linear velocity in [0,1], angular in [-1,1])
    - reward : Sparrow's native reward (+200 win / -200 collide / -200 exceed,
               plus goal-seeking shaping)

Usage:
    python3 train_sparrow_cnn_sac.py
"""
import os
import time

import numpy as np
import torch

from replay_buffer import ReplayBuffer
from sparrow_cspace_env import SparrowCspaceEnv
from cnn_network import CNNSAC
import settings


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
N_ENVS = 32                 # number of vectorized environments (GPU batch)
N_OBSTACLES = 15            # obstacles per env (matches ColorDynamic's default)
batch_size = 256
buffer_size = 250000
file_name = "SAC_sparrow_cspace"
save_model = True
load_model = True
load_model_name = "SAC_sparrow_cspace"

train_every = 5             # env-steps between training calls
gradient_steps = 5          # gradient steps per training call (1:1 with steps)


# ---------------------------------------------------------------------------
# Evaluation (deterministic, vectorized)
# ---------------------------------------------------------------------------
def evaluate(network, env, eval_steps=500):
    """Return (avg_reward, collision_rate) over a vectorized deterministic run."""
    state, _ = env.reset()
    total_reward = 0.0
    num_done = 0
    num_collision = 0
    for _ in range(eval_steps):
        rg = state[:, :settings.RAY_GRID_FLAT_DIM].reshape(
            env.N, 1, settings.RAY_H_BEAMS, settings.RAY_V_LAYERS)
        kin = state[:, settings.RAY_GRID_FLAT_DIM:]
        with torch.no_grad():
            action = network.actor.get_mean_action(rg, kin)      # (N,2) in [-1,1]
        a_in = torch.stack([(action[:, 0] + 1) / 2, action[:, 1]], dim=1)
        state, reward, dw, tr, _ = env.step(a_in)
        done = dw | tr
        total_reward += reward.sum().item()
        num_done += done.sum().item()
        num_collision += env.collide_vec.sum().item()
    avg_reward = total_reward / max(num_done, 1)
    collision_rate = num_collision / max(num_done, 1)
    return avg_reward, collision_rate


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
if save_model and not os.path.exists(settings.MODEL_DIR):
    os.makedirs(settings.MODEL_DIR)
if not os.path.exists(settings.RESULTS_DIR):
    os.makedirs(settings.RESULTS_DIR)

torch.manual_seed(settings.SEED)
np.random.seed(settings.SEED)

env = SparrowCspaceEnv(
    N=N_ENVS,
    O=N_OBSTACLES,
    dvc=settings.device,
    action_type="Continuous",
    render_mode=None,
    max_ep_steps=settings.MAX_STEPS,
)

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
        print("Could not load stored model, starting from random parameters")

print("Training CNN-SAC on Sparrow-V2 c-space state, device=%s, N=%d" %
      (settings.device, N_ENVS))
print("State dim: %d (cost_rays %d + kinematic %d)" %
      (settings.CNN_STATE_DIM, settings.RAY_GRID_FLAT_DIM, settings.KINEMATIC_DIM))

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
state, _ = env.reset()

# per-env episode bookkeeping (Sparrow auto-resets done envs internally)
ep_rewards = torch.zeros(N_ENVS, device=settings.device)
ep_lengths = torch.zeros(N_ENVS, device=settings.device, dtype=torch.long)
logged_episodes = 0

total_steps = 0            # total transitions (env-steps x N)
env_steps = 0              # number of env.step() calls
epoch = 1
evaluations = []
t_start = time.time()

while total_steps < settings.MAX_TIMESTEPS:
    # split flat state into ray-grid + kinematic
    rg = state[:, :settings.RAY_GRID_FLAT_DIM].reshape(
        N_ENVS, 1, settings.RAY_H_BEAMS, settings.RAY_V_LAYERS)
    kin = state[:, settings.RAY_GRID_FLAT_DIM:]

    # sample stochastic action (SAC reparameterized), tanh(-1,1)
    with torch.no_grad():
        action, _ = network.actor.sample(rg, kin)              # (N,2)
    a_in = torch.stack([(action[:, 0] + 1) / 2, action[:, 1]], dim=1)

    next_state, reward, dw, tr, _ = env.step(a_in)
    done = dw | tr

    # store N transitions (t = terminal flag only; truncation bootstraps)
    s_np = state.cpu().numpy()
    a_np = action.cpu().numpy()
    r_np = reward.cpu().numpy()
    t_np = dw.cpu().numpy().astype(np.float32)
    s2_np = next_state.cpu().numpy()
    for i in range(N_ENVS):
        replay_buffer.add(s_np[i], a_np[i], r_np[i], t_np[i], s2_np[i])

    # episode-level logging
    ep_rewards += reward
    ep_lengths += 1
    if done.any():
        for i in done.nonzero(as_tuple=False).squeeze(-1).tolist():
            network.log_episode(
                episode_num=logged_episodes,
                reward=ep_rewards[i].item(),
                length=ep_lengths[i].item(),
                goal_reached=bool(env.win_vec[i].item()),
                collision=bool(env.collide_vec[i].item()),
                expl_noise=0.0,
            )
            logged_episodes += 1
            ep_rewards[i] = 0
            ep_lengths[i] = 0

    state = next_state
    total_steps += N_ENVS
    env_steps += 1

    # training (after warmup), batched every `train_every` env-steps
    if total_steps >= settings.WARMUP_STEPS and env_steps % train_every == 0:
        network.train(replay_buffer, gradient_steps, batch_size,
                      settings.DISCOUNT, settings.TAU)
        if settings.device.type == "cuda":
            torch.cuda.empty_cache()

    # periodic evaluation + save
    if total_steps % settings.EVAL_FREQ == 0:
        avg_reward, avg_col = evaluate(network, env)
        evaluations.append(avg_reward)
        network.log_eval(epoch, avg_reward, avg_col)
        if save_model:
            network.save(file_name, directory=settings.MODEL_DIR)
        np.save("%s/%s" % (settings.RESULTS_DIR, file_name), evaluations)
        print("epoch %d | steps %d | avg_reward %.2f | collision_rate %.3f | "
              "%.1f ksteps/s" % (
                  epoch, total_steps, avg_reward, avg_col,
                  total_steps / 1e3 / max(time.time() - t_start, 1e-6)))
        epoch += 1

# ---------------------------------------------------------------------------
# Final save
# ---------------------------------------------------------------------------
avg_reward, avg_col = evaluate(network, env)
evaluations.append(avg_reward)
network.log_eval(epoch, avg_reward, avg_col)
if save_model:
    network.save(file_name, directory=settings.MODEL_DIR)
np.save("%s/%s" % (settings.RESULTS_DIR, file_name), evaluations)
print("Training complete. Model saved to %s/%s" % (settings.MODEL_DIR, file_name))
