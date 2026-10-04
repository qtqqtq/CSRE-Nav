"""
Break down where wall-clock time goes inside one `SparrowCspaceEnv.step()`.

Why this exists: `train.py` prints "%.1f ksteps/s", which is
N x env.step() calls per second **amortised over the whole loop** (sampling +
network.train + evaluate).  When that number is low on a GPU box it is almost
always the sampling side — env.step() — not the CNN-SAC update.  This script
times the *sub-parts* of one vectorized env.step() so you can see which stage
dominates before deciding what to optimise.

It patches the env's hot methods (plus the cost-ray module functions used inside
`_get_cspace_state`) with timing wrappers, then runs `--steps` vectorized steps
exactly as `train.py` would, reporting mean / median / min
wall-ms and the % of a full step per stage.  CUDA is synchronised at every
boundary so GPU kernel time is counted, not just launch queue.

Usage (inside the container / on your GPU box, from SAC/):
    python3 script/profile_sparrow_step.py --steps 300 [--N 32] [--O 8] [--no-stuck]
    python3 script/profile_sparrow_step.py --backend points --random_footprint True
    # env geometry comes from Sparrow_V2.SPARROW_CSPACE_CFG + settings,
    # i.e. the same grid / footprint / lidar the training script uses.

Stages nest (a stage's time includes whatever it calls, so the column sums can
exceed 100%): `_apply_stuck_reset` re-runs `_get_cspace_state` after moving a
boxed-in robot; `_get_cspace_state` contains, per backend, `_build_occ_grid` /
`build_cspace` / `cost_ray_intervals_3d` (grid) or `_lidar_hits_robot` /
`_pointset_intervals` (points, which itself calls `pointset_batch`), plus
`preprocess_cost_ray_intervals` on both.  Read the %s as "this stage alone
would account for X% of a step if nothing nested", and use the per-stage medians
(the mean is inflated by rare python rejection-sampling loops in `_car_loc_init`
when the map is crowded).

The backend and the footprint regime decide *what the policy sees*, not just
how fast it is produced, so a grid-vs-points timing is only meaningful when
both runs agree on `--N`, `--O`, `--no-stuck` and the thread count.
`torch.set_num_threads` follows `THREADS` from the environment; the same
container has reported 8 and 16 intra-op threads across runs, and that alone
moves the point-set path by ~1.8x.
"""
from __future__ import annotations

import argparse
import collections
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import settings
import Sparrow_V2 as sv2                    # for patching its module globals
from Sparrow_V2 import SparrowCspaceEnv, SPARROW_CSPACE_CFG, str2bool

G = SPARROW_CSPACE_CFG["g_min"]
GMAX = SPARROW_CSPACE_CFG["cost_ray_max_range"]
_GG = 2 * (int(np.ceil(GMAX / G)) + 2) + 1  # c-space grid side length


def _now(dev: torch.device) -> float:
    """perf_counter, flushed so a CUDA launch queue does not hide kernel time."""
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return time.perf_counter()


def patch_methods(env, counters):
    """Wrap env step-path methods + cost-ray module funcs with timing wrappers."""
    sync = env.dvc.type == "cuda"

    def wrap(key, fn):
        def w(*a, **k):
            if sync:
                torch.cuda.synchronize(env.dvc)
                t0 = time.perf_counter()
                r = fn(*a, **k)
                torch.cuda.synchronize(env.dvc)
                dt = time.perf_counter() - t0
            else:
                t0 = time.perf_counter()
                r = fn(*a, **k)
                dt = time.perf_counter() - t0
            counters[key].append(dt)
            return r
        return w

    for name in ["_ld_scan_vec", "_obstacle_move", "_reward_function",
                 "_AutoReset", "_build_occ_grid", "_lidar_hits_robot",
                 "_pointset_intervals", "_get_cspace_state",
                 "_apply_stuck_reset", "_car_loc_init", "_target_point_init",
                 "_reset_env"]:
        fn = getattr(env, name, None)
        if fn is not None:
            setattr(env, name, wrap(name, fn))

    # _get_cspace_state resolves these from the module globals, not as attributes
    for name in ["build_cspace", "cost_ray_intervals_3d", "pointset_batch",
                 "preprocess_cost_ray_intervals"]:
        fn = getattr(sv2, name, None)
        if fn is not None:
            setattr(sv2, name, wrap(name, fn))


def _stat(v, scale=1e3):
    v = np.asarray(v) * scale
    return (float(v.mean()), float(np.median(v)), float(v.min()), int(len(v))) \
        if len(v) else (0.0, 0.0, 0.0, 0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--steps", type=int, default=200, help="vectorized env.step() calls")
    p.add_argument("--N", type=int, default=settings.SPARROW_N_ENVS)
    p.add_argument("--O", type=int, default=settings.SPARROW_N_OBSTACLES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-stuck", action="store_true",
                   help="disable stuck-reset (changes episode dynamics / recompute frequency)")
    p.add_argument("--backend", choices=("grid", "points"),
                   default=settings.CSPACE_BACKEND,
                   help="c-space observation geometry to time; switching it "
                        "changes what the policy sees, so compare backends at "
                        "the same N, O, thread count and footprint regime")
    p.add_argument("--random_footprint", type=str2bool,
                   default=settings.RANDOM_FOOTPRINT_ENABLED,
                   help="draw a new footprint per episode, as train.py does; "
                        "the point-set cost is proportional to the padded "
                        "piece count, so this regime is the slower one")
    args = p.parse_args(argv)

    torch.set_num_threads(int(os.environ.get("THREADS", "0")) or
                          torch.get_num_threads())
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = SparrowCspaceEnv(
        N=args.N, O=args.O, dvc=settings.device,
        action_type="Continuous",
        stuck_reset=not args.no_stuck,
        stuck_threshold=settings.SPARROW_STUCK_THRESHOLD,
        ld_acc=settings.SPARROW_LD_ACC,
        random_footprint=args.random_footprint,
        render_mode=None,                       # sampling-only profile, no pygame
        max_ep_steps=settings.MAX_STEPS,
        cspace_cfg={"backend": args.backend},
    )
    print(f"SparrowCspaceEnv N={env.N} O={env.O} device={env.dvc} "
          f"backend={env.cspace_backend} "
          f"random_footprint={env.random_footprint} "
          f"grid {G}m ({_GG}x{_GG}) stuck_reset={env.stuck_reset} "
          f"ld_acc={env.ld_acc} ld_d_range={env.ld_d_range} "
          f"threads={torch.get_num_threads()}")

    counters = collections.defaultdict(list)
    patch_methods(env, counters)

    state, _ = env.reset()                      # warm-up (also calls _get_cspace_state)
    for _ in range(5):
        a = torch.zeros((env.N, 2), device=env.dvc)
        state, *_ = env.step(a)

    # timed phase
    steps_t = []
    for _ in range(args.steps):
        a = torch.zeros((env.N, 2), device=env.dvc)
        t0 = _now(env.dvc)
        state, *_ = env.step(a)
        steps_t.append(_now(env.dvc) - t0)

    step_mean, step_med, step_min, _ = _stat(steps_t)
    per = {k: _stat(v) for k, v in counters.items()}

    print(f"env.step(): mean {step_mean:.2f} ms  median {step_med:.2f} ms  "
          f"min {step_min:.2f} ms  ->  {args.N / (step_mean / 1e3):.0f} total "
          f"transitions/s (= ksteps/s x N x 1e3)")
    print("(纯采样吞吐，不含 network.train()/evaluate()；训练日志的 ksteps/s 比这个小)\n")

    order = ["_ld_scan_vec", "_get_cspace_state", "_build_occ_grid",
             "build_cspace", "cost_ray_intervals_3d", "_lidar_hits_robot",
             "_pointset_intervals", "pointset_batch",
             "preprocess_cost_ray_intervals", "_obstacle_move",
             "_reward_function", "_AutoReset", "_apply_stuck_reset",
             "_car_loc_init", "_target_point_init", "_reset_env"]
    rows = [(k, *per.get(k, (0, 0, 0, 0))) for k in order if per.get(k, (0, 0, 0, 0))[3]]
    print(f"{'stage':<20}{'calls':>6}{'mean ms':>10}{'median':>10}{'min':>9}{'%of step':>10}")
    for k, m, md, mn, c in rows:
        print(f"{k:<20}{c:>6}{m:>10.3f}{md:>10.3f}{mn:>9.3f}"
              f"{100 * md / max(step_med, 1e-9):>9.1f}%")
    print("\n(% of step 用 median/median 计算，避免个别重采样长尾；嵌套阶段之和可>100%)")

    n_steps = max(len(steps_t), 1)
    obs_calls = per.get("_get_cspace_state", (0, 0, 0, 0))[3]
    stuck_calls = per.get("_apply_stuck_reset", (0, 0, 0, 0))[3]
    # `pointset_batch` runs inside `_pointset_intervals`, so it is deliberately
    # not added again here; the grid geometry stages are siblings and do sum.
    geom_names = (["_build_occ_grid", "build_cspace", "cost_ray_intervals_3d"]
                  if env.cspace_backend == "grid"
                  else ["_lidar_hits_robot", "_pointset_intervals"])
    geom = sum(per.get(k, (0, 0, 0, 0))[0] for k in geom_names)
    obs = per.get("_get_cspace_state", (0, 0, 0, 0))[0]
    print(f"_get_cspace_state 调用数/步 = {obs_calls / n_steps:.2f} "
          f"(stuck 重算会使它 >1) ; _apply_stuck_reset 触发率 = "
          f"{stuck_calls / n_steps:.2f} 次/步")
    print(f"观测 sub-total ({env.cspace_backend}): geometry {geom:.2f} ms "
          f"in _get_cspace_state {obs:.2f} ms (全量 mean, 几何是其子集) "
          f"—— 占 median step 的 {100 * geom / max(step_med, 1e-9):.0f}%")


if __name__ == "__main__":
    raise SystemExit(main())
