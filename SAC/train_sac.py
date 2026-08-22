import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from laser_env import GazeboEnv
import settings


def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        done = False
        while not done and count < 501:
            action = network.get_action(np.array(state))
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


class SACActor(nn.Module):
    """Stochastic Actor for SAC: MLP(400→300→action_dim*2) → (mean, log_std)."""

    def __init__(self, state_dim, action_dim):
        super(SACActor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, action_dim * 2)
        torch.nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="leaky_relu")

    def forward(self, s):
        out = self.layer_3(F.leaky_relu(self.layer_2(F.leaky_relu(self.layer_1(s)))))
        mean, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, settings.LOG_STD_MIN, settings.LOG_STD_MAX)
        return mean, log_std

    def sample(self, s):
        mean, log_std = self.forward(s)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

    def get_mean_action(self, s):
        mean, _ = self.forward(s)
        return torch.tanh(mean)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_s.weight, nonlinearity="leaky_relu")
        self.layer_2_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_a.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="leaky_relu")

        self.layer_4 = nn.Linear(state_dim, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_5_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_s.weight, nonlinearity="leaky_relu")
        self.layer_5_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_a.weight, nonlinearity="leaky_relu")
        self.layer_6 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_6.weight, nonlinearity="leaky_relu")

    def forward(self, s, a):
        s1 = F.leaky_relu(self.layer_1(s))
        s1 = F.leaky_relu(self.layer_2_s(s1) + self.layer_2_a(a))
        q1 = self.layer_3(s1)

        s2 = F.leaky_relu(self.layer_4(s))
        s2 = F.leaky_relu(self.layer_5_s(s2) + self.layer_5_a(a))
        q2 = self.layer_6(s2)
        return q1, q2


# SAC network
class SAC(object):
    """Soft Actor-Critic with auto-tuned temperature alpha."""

    def __init__(self, state_dim, action_dim, max_action, lr=1e-4, alpha_lr=3e-4):
        # Actor (stochastic) — no actor_target in SAC
        self.actor = SACActor(state_dim, action_dim).to(settings.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Critic (twin Q, same structure as TD3)
        self.critic = Critic(state_dim, action_dim).to(settings.device)
        self.critic_target = Critic(state_dim, action_dim).to(settings.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Auto-tune temperature alpha
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=settings.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count = 0

    def get_action(self, state, deterministic=False):
        """Get action from the actor (deterministic for eval, stochastic for train)."""
        state = torch.Tensor(state.reshape(1, -1)).to(settings.device)
        if deterministic:
            return self.actor.get_mean_action(state).cpu().data.numpy().flatten()
        else:
            action, _ = self.actor.sample(state)
            return action.cpu().data.numpy().flatten()

    # training cycle
    def train(self, replay_buffer, iterations, batch_size=100, discount=1, tau=0.005):
        av_Q = 0
        max_Q = -inf
        av_critic_loss = 0
        av_actor_loss = 0
        av_alpha_loss = 0
        av_entropy = 0
        q_soft_targets = []
        q_current1 = []
        q_current2 = []
        for it in range(iterations):
            # sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(settings.device)
            next_state = torch.Tensor(batch_next_states).to(settings.device)
            action = torch.Tensor(batch_actions).to(settings.device)
            reward = torch.Tensor(batch_rewards).to(settings.device)
            done = torch.Tensor(batch_dones).to(settings.device)

            # ---- Critic update ----
            with torch.no_grad():
                next_action, next_log_prob = self.actor.sample(next_state)
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                min_target_Q = torch.min(target_Q1, target_Q2)
                # Soft Bellman backup with entropy bonus
                soft_target = min_target_Q - self.alpha * next_log_prob
                av_Q += torch.mean(soft_target)
                max_Q = max(max_Q, torch.max(soft_target))
                td_target = reward + ((1 - done) * discount * soft_target)
                q_soft_targets.append(soft_target.detach().cpu().numpy())

            current_Q1, current_Q2 = self.critic(state, action)
            q_current1.append(current_Q1.detach().cpu().numpy())
            q_current2.append(current_Q2.detach().cpu().numpy())
            critic_loss = F.mse_loss(current_Q1, td_target) + F.mse_loss(current_Q2, td_target)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # ---- Actor update (every step, reparameterized) ----
            new_action, log_prob = self.actor.sample(state)
            q1_new, q2_new = self.critic(state, new_action)
            min_q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha.detach() * log_prob - min_q_new).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ---- Alpha update (auto-tune temperature) ----
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

            # ---- Soft update critic target only ----
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )

            av_critic_loss += critic_loss.item()
            av_actor_loss += actor_loss.item()
            av_alpha_loss += alpha_loss.item()
            av_entropy += -log_prob.mean().item()

        self.iter_count += 1
        # Write new values for tensorboard
        self.writer.add_scalar("Loss/critic", av_critic_loss / iterations, self.iter_count)
        self.writer.add_scalar("Loss/actor", av_actor_loss / iterations, self.iter_count)
        self.writer.add_scalar("Loss/alpha", av_alpha_loss / iterations, self.iter_count)
        self.writer.add_scalar("Train/alpha", self.alpha.item(), self.iter_count)
        self.writer.add_scalar("Train/entropy", av_entropy / iterations, self.iter_count)
        self.writer.add_scalar("Q/Avg", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Q/Max", max_Q, self.iter_count)
        # Q-value distribution histograms over the training pass
        if q_soft_targets:
            self.writer.add_histogram("Q/dist/soft_target", np.concatenate(q_soft_targets), self.iter_count)
            self.writer.add_histogram("Q/dist/current_Q1", np.concatenate(q_current1), self.iter_count)
            self.writer.add_histogram("Q/dist/current_Q2", np.concatenate(q_current2), self.iter_count)

    def log_episode(self, episode_num, reward, length, goal_reached,
                    collision, expl_noise):
        """Log per-episode training metrics to TensorBoard."""
        self.writer.add_scalar("Episode/reward", reward, episode_num)
        self.writer.add_scalar("Episode/length", length, episode_num)
        self.writer.add_scalar("Episode/goal_reached", int(goal_reached), episode_num)
        self.writer.add_scalar("Episode/collision", int(collision), episode_num)
        self.writer.add_scalar("Episode/alpha", self.alpha.item(), episode_num)

    def log_eval(self, epoch, avg_reward, avg_collision_rate):
        """Log evaluation metrics to TensorBoard."""
        self.writer.add_scalar("Eval/avg_reward", avg_reward, epoch)
        self.writer.add_scalar("Eval/avg_collision_rate", avg_collision_rate, epoch)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )


# Set the script-specific parameters (shared values come from settings.py)
batch_size = 40  # Size of the mini-batch
buffer_size = 1e6  # Maximum size of the buffer
file_name = "SAC_laser_1"  # name of the file to store the policy
save_model = True  # Weather to save the model or not
load_model = False  # Weather to load a stored model (SAC has different arch from TD3)
load_model_name = "SAC_laser"  # name of the stored model to load
random_near_obstacle = False  # To take random actions near obstacles or not

# Create the network storage folders
if not os.path.exists(settings.RESULTS_DIR):
    os.makedirs(settings.RESULTS_DIR)
if save_model and not os.path.exists(settings.MODEL_DIR):
    os.makedirs(settings.MODEL_DIR)

# Create the training environment
env = GazeboEnv("multi_robot_scenario.launch", settings.ENVIRONMENT_DIM,safety_reward_weight=settings.SAFETY_REWARD_WEIGHT,speed_reward_weight=settings.SPEED_REWARD_WEIGHT)
time.sleep(5)
torch.manual_seed(settings.SEED)
np.random.seed(settings.SEED)

# Create the network
network = SAC(settings.STATE_DIM, settings.ACTION_DIM, settings.MAX_ACTION, lr=settings.LR, alpha_lr=settings.ALPHA_LR)
# Create a replay buffer
replay_buffer = ReplayBuffer(buffer_size, settings.SEED)
if load_model:
    try:
        network.load(load_model_name, settings.MODEL_DIR)
    except:
        print(
            "Could not load the stored model parameters, initializing training with random parameters"
        )

# Create evaluation data store
evaluations = []

timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1

count_rand_actions = 0
random_action = []

# Begin the training loop
while timestep < settings.MAX_TIMESTEPS:

    # On termination of episode
    if done:
        # Warm-up: only collect experience into the replay buffer until
        # settings.WARMUP_STEPS total steps have been seen (no gradient updates).
        if timestep >= settings.WARMUP_STEPS:
            network.train(
                replay_buffer,
                episode_timesteps,
                batch_size,
                settings.DISCOUNT,
                settings.TAU,
            )
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
            print("Validating timestep{},episode {}".format(timestep, episode_num))
            timesteps_since_eval %= settings.EVAL_FREQ
            avg_reward, avg_col = evaluate(
                network=network, epoch=epoch, eval_episodes=settings.EVAL_EP)
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

    action = network.get_action(np.array(state))

    # If the robot is facing an obstacle, randomly force it to take a consistent random action.
    # This is done to increase exploration in situations near obstacles.
    # Training can also be performed without it
    if random_near_obstacle:
        if (
            np.random.uniform(0, 1) > 0.85
            and min(state[14:26]) < 0.6
            and count_rand_actions < 1
        ):
            count_rand_actions = np.random.randint(8, 15)
            random_action = np.random.uniform(-1, 1, 2)

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action
            action[0] = -1

    # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)
    done_bool = 0 if episode_timesteps + 1 == settings.MAX_STEPS else int(done)
    done = 1 if episode_timesteps + 1 == settings.MAX_STEPS else int(done)
    episode_reward += reward

    # Save the tuple in replay buffer
    replay_buffer.add(state, action, reward, done_bool, next_state)

    # Update the counters
    state = next_state
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# After the training is done, evaluate the network and save it
avg_reward, avg_col = evaluate(network=network, epoch=epoch, eval_episodes=settings.EVAL_EP)
evaluations.append(avg_reward)
network.log_eval(epoch, avg_reward, avg_col)
if save_model:
    network.save("%s" % file_name, directory=settings.MODEL_DIR)
np.save("%s/%s" % (settings.RESULTS_DIR, file_name), evaluations)
