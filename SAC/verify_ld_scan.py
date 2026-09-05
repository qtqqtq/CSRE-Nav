"""
A/B parity gate for the `Sparrow._ld_scan_vec` occupancy-image rewrite.

`Sparrow_V2.py`'s lidar ray-march (`_ld_scan_vec`) used to test each DDA ray
endpoint against the full obstacle-boundary set:
    goon = ~any(ld_end_code[:,:,None] - vec_bound_code == 0)   # O(N*ld_num*B)
It now gathers from a rasterised occupancy image `base_occ_flat`
(static-contour | map-filled) plus, when dynamic obstacles exist, a small
per-env set test.  The advance / accumulate / car_radius-compensate / ld_GN
grouping arithmetic is unchanged, so outputs must be bit-identical to the old
code.

The old implementation is archived below as `_legacy_scan(env)` (it no longer
lives in Sparrow_V2.py).  Both scans read only car/obstacle geometry and write
the same ld_* tensors, so calling them back-to-back (nothing steps in between)
must produce bit-equal `ld_scan_result` / `ld_result_grouped` / `ld_end_*` /
`ld_angle`.  This script steps each env (exercising the NEW path in the normal
step flow), then on every frozen state runs legacy then new and asserts
torch.equal on all compared tensors.  Passing means the rewrite changes no
downstream value (occ grid, c-space obs, collide_vec, reward).

Usage (inside the container, from SAC/):
    python3 verify_ld_scan.py [--steps 12] [--N 8]
Exits 0 only if every compared tensor is bit-equal on both cases.

Cases:
  (a) cspace env static-only  -- the exact training config (ld_acc=8,
      ld_d_range=300, dynamic obstacles off, stuck_reset on).  Static boundary
      set is large (~38k), i.e. the case that was slow.
  (b) base Sparrow dynamic    -- dynamic_obs=True, O>0, ld_acc=3, ld_d_range=100,
      exercises the has_dyn branch (per-env dynamic set membership still used).
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

import settings
from sparrow_cspace_env import SparrowCspaceEnv
from Sparrow_V2 import Sparrow

# outputs of _ld_scan_vec that downstream code reads; all must be bit-equal
KEYS = ["ld_scan_result", "ld_result_grouped", "ld_vectors_wd",
        "ld_end_wd", "ld_end_gd", "ld_end_code", "ld_angle"]


def _legacy_scan(env):
    """Archived pre-rewrite `_ld_scan_vec` (verbatim, incl. the O(N*ld_num*B)
    membership test against vec_bound_code).  Mutates env's ld_* tensors."""
    self = env
    # 扫描前首先同步雷达与小车位置:
    self.ld_angle = self.ld_angle_interval + self.car_state[:, 2, None]
    self.ld_vectors_wd = torch.stack((torch.cos(self.ld_angle), -torch.sin(self.ld_angle)), dim=2)
    self.ld_end_wd = self.car_state[:, None, 0:2] + self.car_radius * self.ld_vectors_wd
    self.ld_end_gd = self._world_2_grid(self.ld_end_wd)
    self.ld_end_code = self.ld_end_gd[:, :, 0] * self.window_size + self.ld_end_gd[:, :, 1]

    # 扫描初始化
    self.ld_scan_result.fill_(0)
    increment = self.ld_vectors_wd * self.ld_acc

    for i in range(int((self.ld_d_range - self.car_radius) / self.ld_acc) + 2):
        pre_goon = self.ld_end_code[:, :, None] - self.vec_bound_code   # (N,ld_num,1)-(N,1,B)
        goon = ~torch.any(pre_goon == 0, dim=2)
        self.ld_end_wd += (goon[:, :, None] * increment)
        self.ld_end_gd = self._world_2_grid(self.ld_end_wd)
        self.ld_end_code = self.ld_end_gd[:, :, 0] * self.window_size + self.ld_end_gd[:, :, 1]
        self.ld_scan_result += (goon * self.ld_acc)
        if (~goon).all():
            break

    self.ld_scan_result = (self.ld_scan_result + self.car_radius).clamp(0, self.ld_d_range)
    self.ld_result_grouped, _ = torch.min(
        self.ld_scan_result.reshape(self.N, self.grouped_ld_num, self.ld_GN),
        dim=-1, keepdim=False)


def _capture(env):
    return {k: env.__dict__[k].clone() for k in KEYS if k in env.__dict__}


def _compare(env, name: str, tag: str, failures) -> None:
    """run legacy then new on the *current* (frozen) geometry and compare."""
    _legacy_scan(env)                          # legacy writes ld_* from same geometry
    old = _capture(env)
    env._ld_scan_vec()                         # new recomputes from same geometry
    bad = [k for k in old if not torch.equal(old[k], env.__dict__[k])]
    if bad:
        failures.append((name, tag, bad))
        print(f"  [MISMATCH] {name} {tag}: {bad}")
    else:
        print(f"  [ok] {name} {tag}")


def _canary(env, name: str):
    """guard against rasterising the wrong (filled) bitmap: the flat occupancy
    image must contain exactly the static-contour cells that vec_static_bound_code
    encodes (and the map cells when MapObs is set)."""
    img = env.base_occ_flat.cpu()
    st_codes = env.vec_static_bound_code[0, 0].unique().long().cpu()
    assert img[st_codes].all(), f"{name}: a static contour code is missing from base_occ_flat"
    n_st = int(env.static_contour_mask.sum())
    assert n_st == st_codes.numel(), \
        f"{name}: contour-image cells {n_st} != unique static codes {st_codes.numel()}"
    if env.MapObs:
        map_codes = env.vec_map_bound_code[0, 0].unique().long().cpu()
        assert img[map_codes].all(), f"{name}: a map code is missing from base_occ_flat"
    print(f"  [canary] {name}: base_occ_flat occupied={int(img.sum())} "
          f"static_codes={st_codes.numel()}")


def case_cspace(args, failures):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    print("[case (a)] SparrowCspaceEnv static-only (training config)")
    env = SparrowCspaceEnv(
        N=args.N, O=settings.SPARROW_N_OBSTACLES, dvc=torch.device("cpu"),
        action_type="Continuous", render_mode=None,
        ld_acc=settings.SPARROW_LD_ACC,                 # 8
        max_ep_steps=settings.MAX_STEPS,
        stuck_reset=True,
        stuck_threshold=settings.SPARROW_STUCK_THRESHOLD,
    )
    env.reset()
    _canary(env, "cspace")
    _compare(env, "cspace", "after-reset", failures)
    for i in range(args.steps):
        a = (torch.rand((env.N, 2), device=env.dvc) * 2 - 1) * 0.5   # modest moves
        env.step(a)
        _compare(env, "cspace", f"step-{i}", failures)


def case_dynamic(args, failures):
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    print("[case (b)] base Sparrow with dynamic obstacles (has_dyn branch)")
    env = Sparrow(
        N=args.N, O=3, dvc=torch.device("cpu"),
        action_type="Continuous", render_mode=None,
        dynamic_obs=True, ld_acc=3, ld_d_range=100,
    )
    env.reset()
    _canary(env, "dynamic")
    _compare(env, "dynamic", "after-reset", failures)
    for i in range(args.steps):
        env.step(torch.zeros((env.N, 2), device=env.dvc))  # keep car still -> obstacles move
        _compare(env, "dynamic", f"step-{i}", failures)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--steps", type=int, default=12, help="vectorized env.step() per case")
    p.add_argument("--N", type=int, default=8, help="vectorized envs per case")
    args = p.parse_args(argv)

    failures = []
    case_cspace(args, failures)
    case_dynamic(args, failures)

    if failures:
        print(f"\nFAIL: {len(failures)} mismatched scan(s).")
        for name, tag, bad in failures:
            print(f"  - {name} {tag}: {bad}")
        return 1
    print("\nPASS: legacy vs occupancy-image lidar scans are bit-equal on all states.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
