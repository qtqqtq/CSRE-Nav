"""
NavRL-style CNN static obstacle perception network for TD3.

Mirrors NavRL's LiDAR CNN architecture:
  - 3 convolutional layers over the 2D ray-grid (beams × layers)
  - Flatten → Linear(128) → LayerNorm → compact feature vector
  - Actor/Critic heads built on top of CNN features + kinematic state

Reference: NavRL (Xu et al., IEEE RA-L 2025)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# CNN Backbone — Static Obstacle Feature Extractor
# ---------------------------------------------------------------------------

class StaticObstacleCNN(nn.Module):
    """
    CNN that processes a 2D ray-grid into a compact feature vector.

    Architecture (mirrors NavRL):
      Conv2d(C→4,  kernel=[5,3], pad=[2,1]) → ELU
      Conv2d(4→16, kernel=[5,3], stride=[2,1], pad=[2,1]) → ELU
      Conv2d(16→16,kernel=[5,3], stride=[2,2], pad=[2,1]) → ELU
      Flatten → Linear(→feature_dim) → ELU → LayerNorm

    Input shape:  (B, in_channels, H_beams, V_layers)
    Output shape: (B, feature_dim)
    """

    def __init__(self, in_channels=1, h_beams=36, v_layers=4, feature_dim=128):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=[5, 3], padding=[2, 1])
        self.conv2 = nn.Conv2d(4, 16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1])
        self.conv3 = nn.Conv2d(16, 16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1])

        # Compute flattened size by running a dummy forward
        dummy = torch.zeros(1, in_channels, h_beams, v_layers)
        with torch.no_grad():
            d = self.conv1(dummy)
            d = self.conv2(d)
            d = self.conv3(d)
            flat_size = d.numel() // d.shape[0]  # per-sample size

        self.fc = nn.Linear(flat_size, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """x: (B, C, H_beams, V_layers) → features: (B, feature_dim)"""
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = x.flatten(start_dim=1)   # (B, flat_size)
        x = self.fc(x)
        x = F.elu(x)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# CNN-based Actor
# ---------------------------------------------------------------------------

class CNNActor(nn.Module):
    """
    Actor network: CNN ray-grid features + kinematic state → action.

    Pipeline:
      ray_grid (B,1,H,V) → StaticObstacleCNN → (B, feature_dim)
      concat with kinematic (B, kin_dim)
      → MLP [400 → 300 → action_dim] → tanh
    """

    def __init__(self, kinematic_dim=4, action_dim=2,
                 h_beams=36, v_layers=4, cnn_feature_dim=128):
        super().__init__()

        self.cnn = StaticObstacleCNN(
            in_channels=1, h_beams=h_beams, v_layers=v_layers,
            feature_dim=cnn_feature_dim
        )

        input_dim = cnn_feature_dim + kinematic_dim
        self.layer_1 = nn.Linear(input_dim, 400)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity='leaky_relu')
        self.layer_2 = nn.Linear(400, 300)
        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity='leaky_relu')
        self.layer_3 = nn.Linear(300, action_dim)
        nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity='leaky_relu')
        self.tanh = nn.Tanh()

    def forward(self, ray_grid, kinematic):
        """
        Args:
            ray_grid:  (B, 1, H, V)  — preprocessed ray-grid
            kinematic: (B, kin_dim)   — [dist_to_goal, heading_error, lin_vel, ang_vel]
        Returns:
            action: (B, action_dim) in [-1, 1]
        """
        features = self.cnn(ray_grid)
        x = torch.cat([features, kinematic], dim=-1)
        x = F.leaky_relu(self.layer_1(x))
        x = F.leaky_relu(self.layer_2(x))
        a = self.tanh(self.layer_3(x))
        return a


# ---------------------------------------------------------------------------
# CNN-based Twin Critic
# ---------------------------------------------------------------------------

class CNNCritic(nn.Module):
    """
    Twin Critic with a single shared CNN backbone.

    The heavy StaticObstacleCNN is shared between both Q-paths to save GPU
    memory (~50% reduction vs. dual-CNN). The twin structure starts from the
    MLP heads, preserving TD3's double-Q overestimation reduction.

    Pipeline:
      ray_grid → shared CNN → (B, feature_dim)
      concat [features, kinematic] → Q1 MLP [400→300→1]
                                   → Q2 MLP [400→300→1]
    """

    def __init__(self, kinematic_dim=4, action_dim=2,
                 h_beams=36, v_layers=4, cnn_feature_dim=128):
        super().__init__()

        # ---- Shared CNN backbone ----
        self.cnn = StaticObstacleCNN(
            in_channels=1, h_beams=h_beams, v_layers=v_layers,
            feature_dim=cnn_feature_dim
        )

        in_dim = cnn_feature_dim + kinematic_dim

        # ---- Q1 MLP head ----
        self.q1_layer_1 = nn.Linear(in_dim, 400)
        nn.init.kaiming_uniform_(self.q1_layer_1.weight, nonlinearity='leaky_relu')
        self.q1_layer_2_s = nn.Linear(400, 300)
        nn.init.kaiming_uniform_(self.q1_layer_2_s.weight, nonlinearity='leaky_relu')
        self.q1_layer_2_a = nn.Linear(action_dim, 300)
        nn.init.kaiming_uniform_(self.q1_layer_2_a.weight, nonlinearity='leaky_relu')
        self.q1_layer_3 = nn.Linear(300, 1)
        nn.init.kaiming_uniform_(self.q1_layer_3.weight, nonlinearity='leaky_relu')

        # ---- Q2 MLP head (independent weights, shared CNN features) ----
        self.q2_layer_1 = nn.Linear(in_dim, 400)
        nn.init.kaiming_uniform_(self.q2_layer_1.weight, nonlinearity='leaky_relu')
        self.q2_layer_2_s = nn.Linear(400, 300)
        nn.init.kaiming_uniform_(self.q2_layer_2_s.weight, nonlinearity='leaky_relu')
        self.q2_layer_2_a = nn.Linear(action_dim, 300)
        nn.init.kaiming_uniform_(self.q2_layer_2_a.weight, nonlinearity='leaky_relu')
        self.q2_layer_3 = nn.Linear(300, 1)
        nn.init.kaiming_uniform_(self.q2_layer_3.weight, nonlinearity='leaky_relu')

    def forward(self, ray_grid, kinematic, action):
        features = self.cnn(ray_grid)   # shared CNN, computed once
        s = torch.cat([features, kinematic], dim=-1)

        # ---- Q1 ----
        s1 = F.leaky_relu(self.q1_layer_1(s))
        s1 = F.leaky_relu(
            self.q1_layer_2_s(s1) + self.q1_layer_2_a(action))
        q1 = self.q1_layer_3(s1)

        # ---- Q2 ----
        s2 = F.leaky_relu(self.q2_layer_1(s))
        s2 = F.leaky_relu(
            self.q2_layer_2_s(s2) + self.q2_layer_2_a(action))
        q2 = self.q2_layer_3(s2)

        return q1, q2


# ---------------------------------------------------------------------------
# CNN-TD3 — complete algorithm wrapper
# ---------------------------------------------------------------------------

class CNNTD3(object):
    """
    TD3 algorithm using CNN-based Actor and Twin Critic.

    State representation:
      ray_grid  (1, H, V)  — preprocessed ray-grid from point cloud / cost rays
      kinematic (4,)        — [distance_to_goal, heading_error, lin_vel, ang_vel]

    In the replay buffer these are stored as a flat concatenated array.
    """

    def __init__(self, kinematic_dim=4, action_dim=2, max_action=1,
                 h_beams=36, v_layers=4, cnn_feature_dim=128,
                 lr=1e-4, device='cpu',
                 safety_reward_weight=1.0, speed_reward_weight=1.0):
        self.device = device
        self.max_action = max_action
        self.h_beams = h_beams
        self.v_layers = v_layers
        self.kinematic_dim = kinematic_dim
        self.ray_grid_flat_dim = h_beams * v_layers
        self.safety_reward_weight = safety_reward_weight
        self.speed_reward_weight = speed_reward_weight

        # Actor
        self.actor = CNNActor(
            kinematic_dim, action_dim, h_beams, v_layers, cnn_feature_dim
        ).to(device)
        self.actor_target = CNNActor(
            kinematic_dim, action_dim, h_beams, v_layers, cnn_feature_dim
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Critic
        self.critic = CNNCritic(
            kinematic_dim, action_dim, h_beams, v_layers, cnn_feature_dim
        ).to(device)
        self.critic_target = CNNCritic(
            kinematic_dim, action_dim, h_beams, v_layers, cnn_feature_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.writer = SummaryWriter()
        self.iter_count = 0

    def get_action(self, ray_grid, kinematic):
        """
        Get action from the actor.

        Args:
            ray_grid:  numpy (1, H, V) or (H, V)
            kinematic: numpy (kinematic_dim,)
        Returns:
            action: numpy (action_dim,) in [-1, 1]
        """
        rg = torch.Tensor(ray_grid).to(self.device)
        if rg.dim() == 2:
            rg = rg.unsqueeze(0)   # (H,V) → (1,1,H,V)
        elif rg.dim() == 3:
            rg = rg.unsqueeze(0)   # (1,H,V) → (1,1,H,V)
        kn = torch.Tensor(kinematic.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            return self.actor(rg, kn).cpu().data.numpy().flatten()

    def _unpack_batch(self, batch_states):
        """
        Unpack a batch of flat states into ray_grid and kinematic tensors.

        Args:
            batch_states: numpy (B, ray_grid_flat_dim + kinematic_dim)
        Returns:
            ray_grid:  torch (B, 1, H, V)
            kinematic: torch (B, kinematic_dim)
        """
        rg_flat = batch_states[:, :self.ray_grid_flat_dim]
        kin = batch_states[:, self.ray_grid_flat_dim:]
        rg = rg_flat.reshape(-1, 1, self.h_beams, self.v_layers)
        return (
            torch.Tensor(rg).to(self.device),
            torch.Tensor(kin).to(self.device),
        )

    def train(self, replay_buffer, iterations, batch_size=100,
              discount=1, tau=0.005, policy_noise=0.2, noise_clip=0.5,
              policy_freq=2):
        """
        One training pass over `iterations` mini-batches.
        Same interface as the original TD3.train().
        """
        from numpy import inf
        av_Q = 0
        max_Q = -inf
        av_critic_loss = 0
        av_actor_loss = 0
        actor_update_count = 0

        for it in range(iterations):
            # Sample a batch from replay buffer
            (batch_states, batch_actions, batch_rewards,
             batch_dones, batch_next_states) = replay_buffer.sample_batch(batch_size)

            state_rg, state_kin = self._unpack_batch(batch_states)
            next_state_rg, next_state_kin = self._unpack_batch(batch_next_states)
            action = torch.Tensor(batch_actions).to(self.device)
            reward = torch.Tensor(batch_rewards).to(self.device)
            done = torch.Tensor(batch_dones).to(self.device)

            # Target action with noise
            next_action = self.actor_target(next_state_rg, next_state_kin)
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(self.device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Target Q values
            target_Q1, target_Q2 = self.critic_target(
                next_state_rg, next_state_kin, next_action
            )
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Current Q values
            current_Q1, current_Q2 = self.critic(state_rg, state_kin, action)

            # Critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor update (delayed)
            if it % policy_freq == 0:
                actor_loss, _ = self.critic(
                    state_rg, state_kin, self.actor(state_rg, state_kin)
                )
                actor_loss = -actor_loss.mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft update target networks
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

                av_actor_loss += actor_loss.item()
                actor_update_count += 1

            av_critic_loss += critic_loss.item()

        self.iter_count += 1
        self.writer.add_scalar("Loss/critic", av_critic_loss / iterations, self.iter_count)
        self.writer.add_scalar("Loss/actor", av_actor_loss / max(actor_update_count, 1), self.iter_count)
        self.writer.add_scalar("Q/Avg", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Q/Max", max_Q, self.iter_count)

    def log_episode(self, episode_num, reward, length, goal_reached,
                    collision, expl_noise):
        """Log per-episode training metrics to TensorBoard."""
        self.writer.add_scalar("Episode/reward", reward, episode_num)
        self.writer.add_scalar("Episode/length", length, episode_num)
        self.writer.add_scalar("Episode/goal_reached", int(goal_reached), episode_num)
        self.writer.add_scalar("Episode/collision", int(collision), episode_num)
        self.writer.add_scalar("Train/exploration_noise", expl_noise, episode_num)

    def log_eval(self, epoch, avg_reward, avg_collision_rate):
        """Log evaluation metrics to TensorBoard."""
        self.writer.add_scalar("Eval/avg_reward", avg_reward, epoch)
        self.writer.add_scalar("Eval/avg_collision_rate", avg_collision_rate, epoch)

    def log_cost_rays(self, step, cost_rays_proc, cost_rays_raw=None):
        """Visualize cost_rays input as TensorBoard figure (matplotlib-based).

        Avoids PIL.Image compatibility issues in older containers.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Ensure numpy
        if hasattr(cost_rays_proc, 'cpu'):
            cr = cost_rays_proc.detach().cpu().numpy()
        else:
            cr = np.array(cost_rays_proc)
        while cr.ndim > 2:
            cr = cr.squeeze(0)   # → (H, V)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # ── Left: 2D heatmap ──
        im = ax1.imshow(cr.T, aspect='auto', origin='lower',
                        cmap='hot', interpolation='nearest')
        ax1.set_xlabel('beam (0°–360°)')
        ax1.set_ylabel('theta orientation')
        ax1.set_title('CostRays (processed)')
        plt.colorbar(im, ax=ax1, shrink=0.8)

        # ── Right: horizontal profile ──
        beam_mean = cr.mean(axis=-1)
        ax2.plot(range(len(beam_mean)), beam_mean, 'b-', linewidth=1)
        ax2.fill_between(range(len(beam_mean)), 0, beam_mean, alpha=0.3)
        ax2.set_xlabel('beam index')
        ax2.set_ylabel('mean activation')
        ax2.set_title('θ-averaged horizontal profile')
        ax2.set_xlim(0, len(beam_mean) - 1)

        fig.tight_layout()
        self.writer.add_figure("CostRays/processed", fig, step, close=True)

    def save(self, filename, directory):
        """Save actor and critic state dicts."""
        torch.save(self.actor.state_dict(),
                   "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(),
                   "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        """Load actor and critic state dicts."""
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename),
                       map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename),
                       map_location=self.device)
        )
