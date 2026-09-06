"""
NavRL-style CNN static obstacle perception network for SAC.

Mirrors NavRL's LiDAR CNN architecture:
  - 3 convolutional layers over the 2D ray-grid (beams × layers)
  - Flatten → Linear → LayerNorm → compact feature vector
  - SAC Actor (stochastic) / twin Critic heads built on top of the CNN
    features + kinematic state

Reference: NavRL (Xu et al., IEEE RA-L 2025)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from settings import LOG_STD_MIN, LOG_STD_MAX


# ---------------------------------------------------------------------------
# CNN Backbone — Static Obstacle Feature Extractor
# ---------------------------------------------------------------------------

class StaticObstacleCNN(nn.Module):
    """
    CNN that processes a 2D ray-grid into a compact feature vector.

    Architecture (mirrors NavRL, downsampled for the 36×30 cost-ray grid):
      Conv2d(1→4, kernel=[5,3], pad=[2,1], stride=[1,1]) → ELU   (36,30)
      Conv2d(4→8, kernel=[5,3], pad=[2,1], stride=[2,2]) → ELU   (18,15)
      Conv2d(8→8, kernel=[5,3], pad=[2,1], stride=[2,2]) → ELU   ( 9, 8)
      Flatten → Linear(→feature_dim) → ELU → LayerNorm

    Both the beam and the θ axes are halved twice (stride 2), so the 36-beam /
    30-layer cost-ray grid ends up at 9×8 with 8 channels (576-dim) before the
    Linear bottleneck.  Sizes come from a dummy forward, so any input shape
    (h_beams, v_layers) works — no hard-coded dimensions.

    Input shape:  (B, in_channels, H_beams, V_layers)
    Output shape: (B, feature_dim)
    """

    def __init__(self, in_channels=1, h_beams=36, v_layers=4, feature_dim=64):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=[5, 3], padding=[2, 1])
        self.conv2 = nn.Conv2d(4, 8, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1])
        self.conv3 = nn.Conv2d(8, 8, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1])

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
# CNN-based Actor (stochastic policy)
# ---------------------------------------------------------------------------

class CNNActor(nn.Module):
    """
    Actor network (stochastic policy): CNN ray-grid features + kinematic state → (mean, log_std).

    Pipeline:
      ray_grid (B,1,H,V) → StaticObstacleCNN → (B, feature_dim)
      concat with kinematic (B, kin_dim)
      → MLP [400 → 300 → action_dim*2] → (mean, log_std)
    """

    def __init__(self, kinematic_dim=4, action_dim=2,
                 h_beams=36, v_layers=4, cnn_feature_dim=64):
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
        self.layer_3 = nn.Linear(300, action_dim * 2)
        nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity='leaky_relu')

    def forward(self, ray_grid, kinematic):
        """
        Returns:
            mean:      (B, action_dim)
            log_std:   (B, action_dim)  clamped to [LOG_STD_MIN, LOG_STD_MAX]
        """
        features = self.cnn(ray_grid)
        x = torch.cat([features, kinematic], dim=-1)
        x = F.leaky_relu(self.layer_1(x))
        x = F.leaky_relu(self.layer_2(x))
        out = self.layer_3(x)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, ray_grid, kinematic):
        """
        Reparameterization trick with tanh squashing.

        Returns:
            action:    (B, action_dim) in [-1, 1]
            log_prob:  (B, 1)  sum of log-probabilities with tanh correction
        """
        mean, log_std = self.forward(ray_grid, kinematic)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()                       # reparameterization
        action = torch.tanh(x_t)
        # log_prob with tanh squashing correction (sum over action dims)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob

    def get_mean_action(self, ray_grid, kinematic):
        """Deterministic action for evaluation (tanh of mean)."""
        mean, _ = self.forward(ray_grid, kinematic)
        return torch.tanh(mean)


# ---------------------------------------------------------------------------
# CNN-based Twin Critic
# ---------------------------------------------------------------------------

class CNNCritic(nn.Module):
    """
    Twin Critic with a single shared CNN backbone.

    The heavy StaticObstacleCNN is shared between both Q-paths to save GPU
    memory (~50% reduction vs. dual-CNN). The twin structure starts from the
    MLP heads, preserving the double-Q overestimation reduction.

    Pipeline:
      ray_grid → shared CNN → (B, feature_dim)
      concat [features, kinematic] → Q1 MLP [400→300→1]
                                   → Q2 MLP [400→300→1]
    """

    def __init__(self, kinematic_dim=4, action_dim=2,
                 h_beams=36, v_layers=4, cnn_feature_dim=64):
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
# CNN-SAC — Soft Actor-Critic algorithm wrapper
# ---------------------------------------------------------------------------

# 训练多少轮之后才惰性创建 TensorBoard SummaryWriter（runs/ 目录）。writer 的创建
# 以 train() 被调用为门槛：只在真正进入训练、且训练轮数 iter_count 超过该阈值时才写
# runs/。纯推理构造 CNNSAC（play_cspace，只 load + actor 前向，从不调
# train()）因此绝不会产生 runs/ 目录。iter_count 每调用一次 train() 自增 1。
TENSORBOARD_MIN_ITERS = 5


class CNNSAC(object):
    """
    SAC agent combining the CNN-based stochastic Actor and twin Critic.

    The twin-Q structure (a shared CNN backbone feeding two MLP heads) reduces
    the overestimation bias of a single Q-estimator. Only the Actor is
    stochastic; train() runs the standard soft Bellman backup.

    State representation:
      ray_grid  (1, H, V)  — preprocessed ray-grid from point cloud / cost rays
      kinematic (4,)        — [distance_to_goal, heading_error, lin_vel, ang_vel]

    In the replay buffer these are stored as a flat concatenated array.

    TensorBoard: the SummaryWriter is created lazily on the first log/train call
    once train() has run >= tensorboard_min_iters times (see _ensure_writer).
    """

    def __init__(self, kinematic_dim=4, action_dim=2, max_action=1,
                 h_beams=36, v_layers=4, cnn_feature_dim=64,
                 lr=1e-4, alpha_lr=3e-4, device='cpu',
                 safety_reward_weight=1.0, speed_reward_weight=1.0,
                 tensorboard_min_iters=None):
        self.device = device
        self.max_action = max_action
        self.h_beams = h_beams
        self.v_layers = v_layers
        self.kinematic_dim = kinematic_dim
        self.ray_grid_flat_dim = h_beams * v_layers
        self.safety_reward_weight = safety_reward_weight
        self.speed_reward_weight = speed_reward_weight

        # Actor (stochastic) — no actor_target in SAC
        self.actor = CNNActor(
            kinematic_dim, action_dim, h_beams, v_layers, cnn_feature_dim
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Critic (reuses CNNCritic as-is)
        self.critic = CNNCritic(
            kinematic_dim, action_dim, h_beams, v_layers, cnn_feature_dim
        ).to(device)
        self.critic_target = CNNCritic(
            kinematic_dim, action_dim, h_beams, v_layers, cnn_feature_dim
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Auto-tune temperature alpha
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # TensorBoard writer is None until train() has been called
        # tensorboard_min_iters times (then _ensure_writer creates it lazily),
        # so inference-only CNNSAC instances never write a runs/ directory.
        self.writer = None
        self.tensorboard_min_iters = (TENSORBOARD_MIN_ITERS
                                      if tensorboard_min_iters is None
                                      else tensorboard_min_iters)
        self.iter_count = 0

    def _ensure_writer(self):
        """Lazily create the TensorBoard SummaryWriter (runs/ dir).

        Only fires when the network is genuinely being trained (train() ran at
        least self.tensorboard_min_iters times). play_cspace loads a model and
        only runs actor inference — it never calls train(), so its
        CNNSAC keeps writer=None and no runs/ directory is ever created.
        """
        if self.writer is None and self.iter_count >= self.tensorboard_min_iters:
            self.writer = SummaryWriter()

    def get_action(self, ray_grid, kinematic, deterministic=False):
        """
        Get action from the actor.

        Args:
            ray_grid:  numpy (1, H, V) or (H, V)
            kinematic: numpy (kinematic_dim,)
            deterministic: if True, return mean action; else sample stochastically
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
            if deterministic:
                return self.actor.get_mean_action(rg, kn).cpu().data.numpy().flatten()
            else:
                action, _ = self.actor.sample(rg, kn)
                return action.cpu().data.numpy().flatten()

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
              discount=1, tau=0.005):
        """
        One SAC training pass over `iterations` mini-batches.

        SAC algorithm:
          1. Soft Bellman backup: target = r + γ*(min_Q_target - α*log_prob)
          2. Critic loss: MSE(Q - target)
          3. Actor loss: mean(α*log_prob - min_Q)
          4. Alpha loss: -(log_α * (log_prob + target_entropy))
          5. Soft update critic target
        """
        from numpy import inf
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
            # Sample a batch from replay buffer
            (batch_states, batch_actions, batch_rewards,
             batch_dones, batch_next_states) = replay_buffer.sample_batch(batch_size)

            state_rg, state_kin = self._unpack_batch(batch_states)
            next_state_rg, next_state_kin = self._unpack_batch(batch_next_states)
            action = torch.Tensor(batch_actions).to(self.device)
            reward = torch.Tensor(batch_rewards).to(self.device)
            done = torch.Tensor(batch_dones).to(self.device)

            # ---- Critic update ----
            with torch.no_grad():
                next_action, next_log_prob = self.actor.sample(next_state_rg, next_state_kin)
                target_Q1, target_Q2 = self.critic_target(
                    next_state_rg, next_state_kin, next_action
                )
                min_target_Q = torch.min(target_Q1, target_Q2)
                # Soft Bellman backup: subtract entropy bonus
                soft_target = min_target_Q - self.alpha * next_log_prob
                av_Q += torch.mean(soft_target)
                max_Q = max(max_Q, torch.max(soft_target))
                td_target = reward + ((1 - done) * discount * soft_target)
                q_soft_targets.append(soft_target.detach().cpu().numpy())

            current_Q1, current_Q2 = self.critic(state_rg, state_kin, action)
            q_current1.append(current_Q1.detach().cpu().numpy())
            q_current2.append(current_Q2.detach().cpu().numpy())
            critic_loss = F.mse_loss(current_Q1, td_target) + F.mse_loss(current_Q2, td_target)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # ---- Actor update (every step, reparameterized) ----
            new_action, log_prob = self.actor.sample(state_rg, state_kin)
            q1_new, q2_new = self.critic(state_rg, state_kin, new_action)
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

            # ---- Soft update critic target only (no actor target in SAC) ----
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
        # Write new values for tensorboard (writer only exists once training is
        # genuinely underway past the minimum pass count — see _ensure_writer).
        self._ensure_writer()
        if self.writer is None:
            return
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
        """Log per-episode training metrics to TensorBoard.

        No-op before the writer exists (i.e. before training passed the
        tensorboard_min_iters threshold), so warmup episodes are not logged.
        """
        self._ensure_writer()
        if self.writer is None:
            return
        self.writer.add_scalar("Episode/reward", reward, episode_num)
        self.writer.add_scalar("Episode/length", length, episode_num)
        self.writer.add_scalar("Episode/goal_reached", int(goal_reached), episode_num)
        self.writer.add_scalar("Episode/collision", int(collision), episode_num)
        self.writer.add_scalar("Episode/alpha", self.alpha.item(), episode_num)

    def log_eval(self, epoch, avg_reward, avg_collision_rate):
        """Log evaluation metrics to TensorBoard (no-op before writer exists)."""
        self._ensure_writer()
        if self.writer is None:
            return
        self.writer.add_scalar("Eval/avg_reward", avg_reward, epoch)
        self.writer.add_scalar("Eval/avg_collision_rate", avg_collision_rate, epoch)

    def log_cost_rays(self, step, cost_rays_proc, cost_rays_raw=None):
        """Visualize cost_rays input as TensorBoard figure (matplotlib-based).

        Avoids PIL.Image compatibility issues in older containers.
        No-op before the writer exists (training past the min-iter threshold).
        """
        self._ensure_writer()
        if self.writer is None:
            return
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
