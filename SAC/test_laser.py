import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from laser_env import GazeboEnv
import settings


class SACActor(nn.Module):
    """Stochastic Actor for SAC — only mean action used during testing."""

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

    def get_mean_action(self, s):
        mean, _ = self.forward(s)
        return torch.tanh(mean)


# SAC policy wrapper (testing only — deterministic eval)
class SACPolicy(object):
    def __init__(self, state_dim, action_dim):
        self.actor = SACActor(state_dim, action_dim).to(settings.device)

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(settings.device)
        return self.actor.get_mean_action(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )


# Script-specific: name of the file to load the policy from
file_name = "SAC_laser"


# Create the testing environment
env = GazeboEnv("multi_robot_scenario.launch", settings.ENVIRONMENT_DIM)
time.sleep(5)
torch.manual_seed(settings.SEED)
np.random.seed(settings.SEED)

# Create the network
network = SACPolicy(settings.STATE_DIM, settings.ACTION_DIM)
try:
    network.load(file_name, settings.MODEL_DIR)
except:
    raise ValueError("Could not load the stored model parameters")

done = False
episode_timesteps = 0
state = env.reset()

# Begin the testing loop
while True:
    action = network.get_action(np.array(state))

    # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)
    done = 1 if episode_timesteps + 1 == settings.MAX_STEPS else int(done)

    # On termination of episode
    if done:
        state = env.reset()
        done = False
        episode_timesteps = 0
    else:
        state = next_state
        episode_timesteps += 1
