"""Train and compare 1–5 ray intervals with identical training settings.

Run from SAC/ (prefer a CUDA container):
    python3 -u script/run_interval_sweep.py

Each condition collects 2,000,000 transitions and evaluates 200 completed
episodes. Training and test seeds are fixed across conditions. Results are
written under results/interval_sweep_2m/ and model weights under
pytorch_models/interval_sweep_2m/. An interrupted condition restarts from
scratch; successfully trained conditions are not retrained on rerun.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import settings


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=positive_int, default=2_000_000)
    parser.add_argument("--episodes", type=positive_int, default=200)
    parser.add_argument("--buffer-size", type=positive_int, default=80_000)
    parser.add_argument("--train-seed", type=int, default=settings.SEED)
    parser.add_argument("--test-seed", type=int, default=12_345)
    parser.add_argument("--envs", type=positive_int,
                        default=settings.SPARROW_N_ENVS)
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default=str(settings.device))
    parser.add_argument("--model-dir",
                        default="./pytorch_models/interval_sweep_2m")
    parser.add_argument("--results-dir",
                        default="./results/interval_sweep_2m")
    return parser.parse_args()


def run_logged(command, log_path):
    with open(log_path, "w", encoding="utf-8") as log_file:
        subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT,
                       check=True)


def write_summary(rows, results_dir):
    if not rows:
        return
    fields = (
        "intervals", "steps", "episodes", "train_seed", "test_seed",
        "buffer_size", "training_hours", "success_rate", "collision_rate",
        "average_speed_m_s", "stuck_reset_rate", "timeout_rate",
    )
    with open(os.path.join(results_dir, "summary.csv"), "w", newline="",
              encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    rows = []

    for intervals in range(1, 6):
        model_name = "interval_%d_%dsteps" % (intervals, args.steps)
        train_done_path = os.path.join(
            args.results_dir, "%s_train_complete.json" % model_name
        )
        test_json_path = os.path.join(
            args.results_dir, "%s_test.json" % model_name
        )
        actor_path = os.path.join(
            args.model_dir, model_name + "_actor.pth"
        )
        critic_path = os.path.join(
            args.model_dir, model_name + "_critic.pth"
        )

        if not (os.path.isfile(train_done_path)
                and os.path.isfile(actor_path)
                and os.path.isfile(critic_path)):
            print("Training %d intervals for %d steps..." % (
                intervals, args.steps), flush=True)
            start = time.time()
            train_command = [
                sys.executable, "-u", "train.py",
                "--intervals", str(intervals),
                "--steps", str(args.steps),
                "--name", model_name,
                "--model-dir", args.model_dir,
                "--results-dir", args.results_dir,
                "--buffer-size", str(args.buffer_size),
                "--seed", str(args.train_seed),
            ]
            run_logged(train_command, os.path.join(
                args.results_dir, "%s_train.log" % model_name
            ))
            with open(train_done_path, "w", encoding="utf-8") as done_file:
                json.dump({
                    "intervals": intervals,
                    "steps": args.steps,
                    "buffer_size": args.buffer_size,
                    "train_seed": args.train_seed,
                    "training_hours": (time.time() - start) / 3600.0,
                }, done_file, indent=2)

        if not os.path.isfile(test_json_path):
            print("Testing %d intervals for %d episodes..." % (
                intervals, args.episodes), flush=True)
            test_command = [
                sys.executable, "-u", "script/test_cspace.py",
                "--model", model_name,
                "--model-dir", args.model_dir,
                "--intervals", str(intervals),
                "--episodes", str(args.episodes),
                "--envs", str(args.envs),
                "--device", args.device,
                "--seed", str(args.test_seed),
                "--json-out", test_json_path,
            ]
            run_logged(test_command, os.path.join(
                args.results_dir, "%s_test.log" % model_name
            ))

        with open(train_done_path, encoding="utf-8") as done_file:
            train_info = json.load(done_file)
        with open(test_json_path, encoding="utf-8") as test_file:
            test_info = json.load(test_file)
        row = {
            "intervals": intervals,
            "steps": train_info["steps"],
            "episodes": test_info["episodes"],
            "train_seed": train_info["train_seed"],
            "test_seed": test_info["seed"],
            "buffer_size": train_info["buffer_size"],
            "training_hours": train_info["training_hours"],
            "success_rate": test_info["success_rate"],
            "collision_rate": test_info["collision_rate"],
            "average_speed_m_s": test_info["average_speed_m_s"],
            "stuck_reset_rate": test_info["stuck_reset_rate"],
            "timeout_rate": test_info["timeout_rate"],
        }
        rows.append(row)
        write_summary(rows, args.results_dir)
        print("%d intervals: success %.1f%%, collision %.1f%%, speed %.3f m/s" % (
            intervals, 100 * row["success_rate"],
            100 * row["collision_rate"], row["average_speed_m_s"],
        ), flush=True)

    print("Comparison saved to %s/summary.csv" % args.results_dir, flush=True)


if __name__ == "__main__":
    main()
