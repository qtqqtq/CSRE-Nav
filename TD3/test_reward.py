"""
Reward function test script — runs in real Gazebo with three action policies.

Scenarios:
  1. Random actions — uniform random [linear, angular]
  2. Straight-to-goal — proportional controller driving directly toward the goal
  3. Trained TD3 model — loads a saved actor network for inference

Each scenario runs for one episode (max 500 steps) and logs per-step reward
components: r_vel (velocity alignment), r_ss (log-average static safety),
distance to goal, and min laser reading.

Usage:
  python3 test_reward.py
"""

import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from velodyne_env import GazeboEnv

# ---------------------------------------------------------------------------
# Network definitions (must match train_velodyne_td3.py exactly)
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.leaky_relu(self.layer_1(s))
        s = F.leaky_relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class TD3Policy(object):
    """Minimal wrapper — only the actor, for inference."""
    def __init__(self, state_dim, action_dim, device):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.device = device

    def get_action(self, state):
        state_t = torch.Tensor(state.reshape(1, -1)).to(self.device)
        a = self.actor(state_t).cpu().data.numpy().flatten()
        return a

    def load(self, filename, directory):
        path = os.path.join(directory, "%s_actor.pth" % filename)
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
        print("  Loaded: %s" % path)


# ---------------------------------------------------------------------------
# Action policies
# ---------------------------------------------------------------------------

def random_action():
    """Uniform random action in the valid range."""
    return np.array([np.random.uniform(0.0, 1.0),
                     np.random.uniform(-1.0, 1.0)])


def straight_to_goal_action(env):
    """Proportional controller driving straight toward the goal.

    Reads current odometry and goal position from the environment.
    Uses proportional control for angular velocity and distance-based
    linear velocity.  No laser feedback — purely reactive to goal pose.
    """
    from squaternion import Quaternion

    odom = env.last_odom
    if odom is None:
        return np.array([0.0, 0.0])

    dx = env.goal_x - env.odom_x
    dy = env.goal_y - env.odom_y
    distance = math.sqrt(dx**2 + dy**2)

    q = Quaternion(
        odom.pose.pose.orientation.w,
        odom.pose.pose.orientation.x,
        odom.pose.pose.orientation.y,
        odom.pose.pose.orientation.z,
    )
    yaw = q.to_euler(degrees=False)[2]
    goal_angle = math.atan2(dy, dx)
    theta = math.atan2(math.sin(goal_angle - yaw), math.cos(goal_angle - yaw))

    Kp = 1.5
    angular = np.clip(theta * Kp, -1.0, 1.0)
    linear = np.clip(distance * 0.5, 0.0, 1.0)
    linear *= (1.0 - abs(angular) * 0.5)  # slow down when turning

    return np.array([linear, angular])


# ---------------------------------------------------------------------------
# Reward decomposition helper
# ---------------------------------------------------------------------------

def decompose_reward(env, action, reward_total):
    """Re-compute reward components from env state for logging."""
    from velodyne_env import GOAL_REACHED_DIST, COLLISION_DIST

    odom = env.last_odom
    if odom is None:
        return {"total": reward_total}

    odom_x = odom.pose.pose.position.x
    odom_y = odom.pose.pose.position.y

    from squaternion import Quaternion
    q = Quaternion(
        odom.pose.pose.orientation.w,
        odom.pose.pose.orientation.x,
        odom.pose.pose.orientation.y,
        odom.pose.pose.orientation.z,
    )
    heading_angle = q.to_euler(degrees=False)[2]

    distance = math.sqrt((odom_x - env.goal_x)**2 + (odom_y - env.goal_y)**2)
    min_laser = min(env.velodyne_data)
    target = distance < GOAL_REACHED_DIST
    collision = min_laser < COLLISION_DIST

    # r_vel: velocity alignment toward goal
    vx = action[0] * math.cos(heading_angle)
    vy = action[0] * math.sin(heading_angle)
    dx_g = env.goal_x - odom_x
    dy_g = env.goal_y - odom_y
    dist_to_goal = math.sqrt(dx_g**2 + dy_g**2) + 1e-8
    r_vel = (dx_g / dist_to_goal) * vx + (dy_g / dist_to_goal) * vy

    # r_ss: log-average static safety
    clipped = np.maximum(np.array(env.velodyne_data), 0.1)
    r_ss = np.mean(np.log(clipped))

    return {
        "total": reward_total,
        "r_vel": r_vel,
        "r_ss": r_ss,
        "distance": distance,
        "min_laser": min_laser,
        "target": target,
        "collision": collision,
        "goal_x": env.goal_x,
        "goal_y": env.goal_y,
        "odom_x": odom_x,
        "odom_y": odom_y,
        "heading": heading_angle,
    }


# ---------------------------------------------------------------------------
# Episode runner (for env-based policies: random, straight-to-goal)
# ---------------------------------------------------------------------------

def run_episode_env(env, policy_fn, policy_name, max_steps=500):
    """Run one episode.  policy_fn(env) → action."""
    print("\n" + "=" * 70)
    print("Running: %s" % policy_name)
    print("=" * 70)

    env.reset()
    total_reward = 0.0
    reward_log = []

    for step_idx in range(max_steps):
        action = policy_fn(env)
        _, reward, done, target = env.step(action)

        comps = decompose_reward(env, action, reward)
        reward_log.append(comps)
        total_reward += reward

        if step_idx < 5 or step_idx % 50 == 0 or done:
            tag = "GOAL!" if target else ("COLLISION!" if comps["collision"] else "")
            print("  step %3d | a=[%.2f, %+.2f] | r=%+.2f | "
                  "r_vel=%+.3f r_ss=%+.3f | dist=%.2f | laser_min=%.2f | %s" % (
                      step_idx, action[0], action[1],
                      reward, comps["r_vel"], comps["r_ss"],
                      comps["distance"], comps["min_laser"], tag))

        if done:
            print("  → Ended step %d: goal=%s collision=%s" % (
                step_idx + 1, target, comps["collision"]))
            break

    return _build_result(policy_name, reward_log, total_reward)


# ---------------------------------------------------------------------------
# Episode runner for state-based policy (TD3)
# ---------------------------------------------------------------------------

def run_episode_state(env, policy, policy_name, max_steps=500):
    """Run one episode.  policy.get_action(state) → raw tanh action."""
    print("\n" + "=" * 70)
    print("Running: %s" % policy_name)
    print("=" * 70)

    state = env.reset()
    total_reward = 0.0
    reward_log = []

    for step_idx in range(max_steps):
        action_raw = policy.get_action(np.array(state))
        # Map tanh output: linear [-1,1]→[0,1], angular stays [-1,1]
        action = np.array([(action_raw[0] + 1) / 2, action_raw[1]])
        next_state, reward, done, target = env.step(action)

        comps = decompose_reward(env, action, reward)
        reward_log.append(comps)
        total_reward += reward

        if step_idx < 5 or step_idx % 50 == 0 or done:
            tag = "GOAL!" if target else ("COLLISION!" if comps["collision"] else "")
            print("  step %3d | a=[%.2f, %+.2f] | r=%+.2f | "
                  "r_vel=%+.3f r_ss=%+.3f | dist=%.2f | laser_min=%.2f | %s" % (
                      step_idx, action[0], action[1],
                      reward, comps["r_vel"], comps["r_ss"],
                      comps["distance"], comps["min_laser"], tag))

        if done:
            print("  → Ended step %d: goal=%s collision=%s" % (
                step_idx + 1, target, comps["collision"]))
            break

        state = next_state

    return _build_result(policy_name, reward_log, total_reward)


def _build_result(policy_name, reward_log, total_reward):
    goal_reached = reward_log[-1]["target"] if reward_log else False
    collision = reward_log[-1]["collision"] if reward_log else False
    return {
        "policy": policy_name,
        "steps": len(reward_log),
        "total_reward": total_reward,
        "goal_reached": goal_reached,
        "collision": collision,
        "rewards": reward_log,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = "%-25s %6s %10s %10s %8s %10s %8s %8s %8s" % (
        "Policy", "Steps", "Total R", "Avg R", "Goal?", "Collision?",
        "r_vel μ", "r_ss μ", "r_ss min")
    print(header)
    print("-" * len(header))
    for r in results:
        if r is None:
            continue
        n = max(r["steps"], 1)
        avg = r["total_reward"] / n
        rv = np.mean([d["r_vel"] for d in r["rewards"]]) if r["rewards"] else 0
        rs = np.mean([d["r_ss"] for d in r["rewards"]]) if r["rewards"] else 0
        rs_min = np.min([d["r_ss"] for d in r["rewards"]]) if r["rewards"] else 0
        print("%-25s %6d %+10.2f %10.2f %8s %10s %+8.3f %8.3f %8.3f" % (
            r["policy"], r["steps"], r["total_reward"], avg,
            "YES" if r["goal_reached"] else "no",
            "YES" if r["collision"] else "no",
            rv, rs, rs_min))

    print("\nReward component distributions:")
    for r in results:
        if r is None or not r["rewards"]:
            continue
        totals = np.array([d["total"] for d in r["rewards"]])
        rv_arr = np.array([d["r_vel"] for d in r["rewards"]])
        rs_arr = np.array([d["r_ss"] for d in r["rewards"]])
        print("  %s:" % r["policy"])
        print("    total: μ=%+.3f  σ=%.3f  min=%+.3f  max=%+.3f" % (
            totals.mean(), totals.std(), totals.min(), totals.max()))
        print("    r_vel: μ=%+.3f  σ=%.3f  min=%+.3f  max=%+.3f" % (
            rv_arr.mean(), rv_arr.std(), rv_arr.min(), rv_arr.max()))
        print("    r_ss:  μ=%+.3f  σ=%.3f  min=%+.3f  max=%+.3f" % (
            rs_arr.mean(), rs_arr.std(), rs_arr.min(), rs_arr.max()))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ENVIRONMENT_DIM = 40
    ROBOT_DIM = 4
    STATE_DIM = ENVIRONMENT_DIM + ROBOT_DIM  # 44
    ACTION_DIM = 2
    MODEL_NAME = "TD3_velodyne_360"
    MODEL_DIR = "./pytorch_models"
    MAX_STEPS = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s  |  State dim: %d  |  Action dim: %d" % (
        device, STATE_DIM, ACTION_DIM))

    # --- Create environment ---
    print("\nLaunching Gazebo environment ...")
    env = GazeboEnv("multi_robot_scenario.launch", ENVIRONMENT_DIM)
    time.sleep(5)

    # --- Load trained model ---
    policy_td3 = None
    model_path = os.path.join(MODEL_DIR, "%s_actor.pth" % MODEL_NAME)
    if os.path.exists(model_path):
        policy_td3 = TD3Policy(STATE_DIM, ACTION_DIM, device)
        try:
            policy_td3.load(MODEL_NAME, MODEL_DIR)
        except Exception as e:
            print("  WARNING: Failed to load model: %s" % e)
            policy_td3 = None
    else:
        print("WARNING: Model not found at %s — skipping TD3 scenario" % model_path)

    results = []

    # --- Scenario 1: Random actions ---
    results.append(run_episode_env(env, lambda e: random_action(),
                                   "Random Actions", MAX_STEPS))

    # --- Scenario 2: Straight-to-goal ---
    results.append(run_episode_env(env, lambda e: straight_to_goal_action(e),
                                   "Straight-to-Goal", MAX_STEPS))

    # --- Scenario 3: Trained TD3 model ---
    if policy_td3 is not None:
        results.append(run_episode_state(env, policy_td3,
                                         "TD3 (%s)" % MODEL_NAME, MAX_STEPS))

    # --- Summary ---
    print_summary(results)
    print("\nDone.")
