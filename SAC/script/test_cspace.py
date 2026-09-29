"""Evaluate a trained CNN-SAC actor in the same Sparrow environment as train.py.

Run from SAC/:
    python3 script/test_cspace.py --model interval_cspace_1 --episodes 100

Success means reaching the goal. Collision rate is the fraction of completed
episodes with at least one lidar-detected contact (including a stuck reset).
Stuck-reset rate is also printed for comparison with train.py's eval metric.
Average speed is total absolute linear travel divided by elapsed simulated time
over completed episodes, in m/s; unfinished episodes are excluded.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import settings
from Sparrow_V2 import SparrowCspaceEnv
from cnn_network import CNNActor


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="model basename without the _actor.pth suffix")
    parser.add_argument("--model-dir", default=settings.MODEL_DIR,
                        help="directory containing the actor checkpoint")
    parser.add_argument("--intervals", type=positive_int,
                        default=settings.CSPACE_COST_RAY_NUM_INTERVALS,
                        help="ray interval channels used when training the model")
    parser.add_argument("--episodes", type=positive_int, default=100,
                        help="number of completed navigation episodes (default: 100)")
    parser.add_argument("--envs", type=positive_int,
                        default=settings.SPARROW_N_ENVS,
                        help="parallel environments (default: training setting)")
    parser.add_argument("--device", default=str(settings.device),
                        choices=("cpu", "cuda"), help="evaluation device")
    parser.add_argument("--seed", type=int, default=settings.SEED)
    parser.add_argument("--json-out", help="optional path for machine-readable results")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    model_path = os.path.join(args.model_dir, args.model + "_actor.pth")
    if not os.path.isfile(model_path):
        raise FileNotFoundError("Actor checkpoint not found: %s" % model_path)

    actor = CNNActor(
        kinematic_dim=settings.KINEMATIC_DIM,
        action_dim=settings.ACTION_DIM,
        h_beams=settings.RAY_H_BEAMS,
        v_layers=settings.RAY_V_LAYERS,
        ray_channels=args.intervals,
        cnn_feature_dim=64,
    ).to(device)
    actor.load_state_dict(torch.load(
        model_path, map_location=device, weights_only=True,
    ))
    actor.eval()

    env_count = min(args.envs, args.episodes)
    ray_grid_flat_dim = (
        args.intervals * settings.RAY_H_BEAMS * settings.RAY_V_LAYERS
    )
    env = SparrowCspaceEnv(
        N=env_count,
        O=settings.SPARROW_N_OBSTACLES,
        dvc=device,
        action_type="Continuous",
        stuck_reset=settings.SPARROW_STUCK_RESET,
        stuck_threshold=settings.SPARROW_STUCK_THRESHOLD,
        ld_acc=settings.SPARROW_LD_ACC,
        global_plan=settings.GLOBAL_PLAN_ENABLED,
        render_mode=None,
        max_ep_steps=settings.MAX_STEPS,
        cspace_cfg={"cost_ray_num_intervals": args.intervals},
    )

    completed = successes = collisions = stuck_resets = timeouts = 0
    distance_m = 0.0
    elapsed_s = 0.0
    episode_distance = torch.zeros(env_count, device=device)
    episode_time = torch.zeros(env_count, device=device)
    episode_contact = torch.zeros(env_count, dtype=torch.bool, device=device)

    try:
        # Actor initialization consumes a channel-dependent number of random
        # draws. Reseed immediately before reset so all interval counts see the
        # same initial test worlds for a given seed.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        state, _ = env.reset()
        with torch.inference_mode():
            while completed < args.episodes:
                ray_grid = state[:, :ray_grid_flat_dim].reshape(
                    env_count, args.intervals,
                    settings.RAY_H_BEAMS, settings.RAY_V_LAYERS,
                )
                kinematic = state[:, ray_grid_flat_dim:]
                action = actor.get_mean_action(ray_grid, kinematic)
                state, _, terminated, truncated, _ = env.step(
                    settings.map_continuous_action(action)
                )

                dt = env.ctrl_interval.reshape(-1)  # seconds, one per env
                episode_distance += env.car_state[:, 3].abs() * dt / 100.0
                episode_time += dt
                # Ordinary lidar contacts do not terminate SparrowCspaceEnv.
                # Remember whether each episode touched an obstacle at any step.
                contact = (env.ld_result_grouped < env.collision_trsd).any(-1)
                episode_contact |= contact | env.collide_vec

                done = terminated | truncated
                for index in done.nonzero(as_tuple=False).flatten().tolist():
                    if completed == args.episodes:
                        break
                    successes += int(env.win_vec[index].item())
                    collisions += int(episode_contact[index].item())
                    stuck_resets += int(env.collide_vec[index].item())
                    timeouts += int(truncated[index].item())
                    distance_m += episode_distance[index].item()
                    elapsed_s += episode_time[index].item()
                    completed += 1
                    episode_distance[index] = 0
                    episode_time[index] = 0
                    episode_contact[index] = False
    finally:
        env.close()

    metrics = {
        "model": args.model,
        "intervals": args.intervals,
        "device": str(device),
        "seed": args.seed,
        "episodes": completed,
        "envs": env_count,
        "successes": successes,
        "collisions": collisions,
        "stuck_resets": stuck_resets,
        "timeouts": timeouts,
        "success_rate": successes / completed,
        "collision_rate": collisions / completed,
        "stuck_reset_rate": stuck_resets / completed,
        "timeout_rate": timeouts / completed,
        "average_speed_m_s": distance_m / elapsed_s,
    }
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as result_file:
            json.dump(metrics, result_file, indent=2)

    print("model: %s | intervals: %d | device: %s | seed: %d | episodes: %d | envs: %d" % (
        args.model, args.intervals, device, args.seed, completed, env_count,
    ))
    print("success rate: %.2f%% (%d/%d)" % (
        100.0 * successes / completed, successes, completed,
    ))
    print("collision rate: %.2f%% (%d/%d episodes with contact)" % (
        100.0 * collisions / completed, collisions, completed,
    ))
    print("average speed: %.3f m/s (%.1f cm/s)" % (
        distance_m / elapsed_s, 100.0 * distance_m / elapsed_s,
    ))
    print("stuck-reset rate (train.py metric): %.2f%% (%d/%d)" % (
        100.0 * stuck_resets / completed, stuck_resets, completed,
    ))
    print("timeout rate: %.2f%% (%d/%d)" % (
        100.0 * timeouts / completed, timeouts, completed,
    ))


if __name__ == "__main__":
    main()
