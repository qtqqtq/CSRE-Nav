"""
Verify cspace_cost_rays against hand-computed and naive-reference values.

Three independent checks (run inside the container):

    python SAC/verify_cost_rays.py

  A. Analytic      : identity footprint + single obstacle cells -> hand-derived
                     first-hit distances (physical near-edge of the occupied cell).
  B. cost_rays_3d  : vs a naive per-(env, beam, theta, step) loop that re-derives
                     ci/cj/ck with the SAME scalar arithmetic but no vectorized
                     gather/expand/active-masking.  Must match EXACTLY.
  C. build_cspace  : vs a naive per-cell footprint-overlap (dilation) loop.  Must
                     match EXACTLY (catches any conv2d transpose/padding bug).
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from cspace_cost_rays import (
    make_footprint_masks,
    build_cspace,
    cost_rays_3d,
    compute_cost_rays,
)

# Footprints (metres).  Identity square < 1 cell => dilation == identity.
IDENTITY_FOOTPRINT = [[0.05, 0.05], [0.05, -0.05], [-0.05, -0.05], [-0.05, 0.05]]
CAR_FOOTPRINT = [[0.20, 0.06], [0.20, -0.06], [-0.16, -0.10], [-0.20, 0.00], [-0.16, 0.10]]

A = 0.698          # theta resolution (rad) -> nz = 10
NZ = 10
BEAMS = 36
H_RES = 10.0


def _cell_to_world(offset, g_min):
    """World distance (m) of a cell that is `offset` cells from the robot centre."""
    return offset * g_min


def _near_edge(offset, g_min):
    """Robot-centre distance at which the ray first touches cell `offset`."""
    return (offset - 0.5) * g_min


# ---------------------------------------------------------------------------
# Naive references (independent of the vectorized implementations)
# ---------------------------------------------------------------------------
def naive_cost_rays(cspace, robot_cell, num_beams, h_res_deg, max_range, g_min, a,
                    acc=None):
    N, nx, ny, nz = cspace.shape
    dev = cspace.device
    acc = g_min / 2.0 if acc is None else acc
    n_steps = int(math.ceil(max_range / acc)) + 2
    out = torch.full((N, num_beams, nz), max_range, device=dev)
    beam_ang = torch.arange(num_beams, device=dev).float() * math.radians(h_res_deg)
    k_idx = torch.arange(nz, device=dev).float()
    for n in range(N):
        ri = robot_cell[n, 0].float()
        rj = robot_cell[n, 1].float()
        for b in range(num_beams):
            dx = torch.cos(beam_ang[b])
            dy = torch.sin(beam_ang[b])
            for k in range(nz):
                d_theta = (k_idx[k] * a) / max_range
                for s in range(1, n_steps + 1):
                    t = s * acc
                    ci = (ri + torch.round(dx * t / g_min)).long().item()
                    cj = (rj + torch.round(dy * t / g_min)).long().item()
                    ck = torch.floor(d_theta * t / a).long().item()
                    ck = min(max(ck, 0), nz - 1)
                    if 0 <= ci < nx and 0 <= cj < ny and cspace[n, ci, cj, ck]:
                        out[n, b, k] = t
                        break
    return out


def naive_build_cspace(occ, masks):
    N, nx, ny = occ.shape
    nz, m, _ = masks.shape
    half = m // 2
    out = torch.zeros(N, nx, ny, nz, dtype=torch.bool, device=occ.device)
    for n in range(N):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    hit = False
                    for mi in range(m):
                        for mj in range(m):
                            if masks[k, mi, mj] > 0.5:
                                oi = i + mi - half
                                oj = j + mj - half
                                if 0 <= oi < nx and 0 <= oj < ny and occ[n, oi, oj]:
                                    hit = True
                                    break
                        if hit:
                            break
                    out[n, i, j, k] = hit
    return out


# ---------------------------------------------------------------------------
# Section A — analytic (identity footprint)
# ---------------------------------------------------------------------------
def check_A():
    g_min = 0.1
    max_range = 4.0
    R = int(math.ceil(max_range / g_min)) + 2
    G = 2 * R + 1
    robot_cell = torch.tensor([[R, R]], dtype=torch.long)

    occ = torch.zeros(1, G, G, dtype=torch.bool)
    occ[0, R + 5, R] = True     # 0.5 m ahead  (+x, beam 0)
    occ[0, R, R + 10] = True    # 1.0 m left   (+y, beam 9)

    cfg = dict(g_min=g_min, a=A, nz=NZ, footprint=IDENTITY_FOOTPRINT,
               cost_ray_max_range=max_range, cost_ray_num_beams=BEAMS,
               cost_ray_h_res_degrees=H_RES)
    cr = compute_cost_rays(occ, robot_cell, cfg)              # (1, 36, 10)

    # identity footprint => c-space == occ (broadcast over theta)
    masks = torch.as_tensor(make_footprint_masks(g_min, A, NZ, IDENTITY_FOOTPRINT),
                            dtype=torch.float32)
    cspace = build_cspace(occ, masks)                          # (1, G, G, 10)
    assert bool(torch.all(cspace == occ.unsqueeze(-1))), "identity footprint must not dilate"

    ok = True
    checks = [
        ("beam 0  (+x) hits cell offset +5", cr[0, 0, 0].item(), _near_edge(5, g_min)),
        ("beam 9  (+y) hits cell offset +10", cr[0, 9, 0].item(), _near_edge(10, g_min)),
        ("beam 18 (-x) empty", cr[0, 18, 0].item(), max_range),
        ("beam 27 (-y) empty", cr[0, 27, 0].item(), max_range),
    ]
    for name, got, exp in checks:
        if name.endswith("empty"):
            passed = abs(got - exp) < 1e-6
        else:
            # within one cell (g_min) of the physical near-edge
            passed = abs(got - exp) <= g_min + 1e-6
        ok &= passed
        print("  [A] %-30s got=%7.3f  expect~%7.3f  %s"
              % (name, got, exp, "OK" if passed else "FAIL"))
    return ok


# ---------------------------------------------------------------------------
# Section B — cost_rays_3d vs naive loop (anisotropic footprint, full grid)
# ---------------------------------------------------------------------------
def check_B():
    g_min = 0.1
    max_range = 4.0
    R = int(math.ceil(max_range / g_min)) + 2
    G = 2 * R + 1
    robot_cell = torch.tensor([[R, R]], dtype=torch.long)

    masks = torch.as_tensor(make_footprint_masks(g_min, A, NZ, CAR_FOOTPRINT),
                            dtype=torch.float32)
    cspace = build_cspace(torch.zeros(1, G, G, dtype=torch.bool), masks)  # dummy

    # hand-placed scene: a wall + a blob
    occ = torch.zeros(1, G, G, dtype=torch.bool)
    occ[0, R + 15, R - 5:R + 6] = True           # vertical wall 1.5 m ahead
    occ[0, R + 3, R + 18:R + 22] = True          # small blob up-left

    cspace = build_cspace(occ, masks)            # (1, G, G, NZ)
    got = cost_rays_3d(cspace, robot_cell, num_beams=BEAMS, h_res_deg=H_RES,
                       max_range=max_range, g_min=g_min, a=A)
    ref = naive_cost_rays(cspace, robot_cell, num_beams=BEAMS, h_res_deg=H_RES,
                          max_range=max_range, g_min=g_min, a=A)

    max_err = (got - ref).abs().max().item()
    n_diff = int((got != ref).sum().item())
    ok = (max_err == 0.0)
    print("  [B] cost_rays_3d vs naive: max_abs_err=%.3g  mismatches=%d  %s"
          % (max_err, n_diff, "OK" if ok else "FAIL"))
    if not ok:
        idx = (got != ref).nonzero()
        for t in idx[:10]:
            print("       first mismatches at", t.tolist(),
                  "got", got[tuple(t)].item(), "ref", ref[tuple(t)].item())
    return ok


# ---------------------------------------------------------------------------
# Section C — build_cspace vs naive dilation (small grid, anisotropic footprint)
# ---------------------------------------------------------------------------
def check_C():
    g_min = 0.1
    max_range = 0.5                       # small grid for a fast exhaustive loop
    R = int(math.ceil(max_range / g_min)) + 2
    G = 2 * R + 1
    masks = torch.as_tensor(make_footprint_masks(g_min, A, NZ, CAR_FOOTPRINT),
                            dtype=torch.float32)

    occ = torch.zeros(1, G, G, dtype=torch.bool)
    occ[0, R + 2, R] = True
    occ[0, R, R - 2] = True

    got = build_cspace(occ, masks)          # (1, G, G, NZ)
    ref = naive_build_cspace(occ, masks)    # (1, G, G, NZ)

    n_diff = int((got != ref).sum().item())
    ok = (n_diff == 0)
    print("  [C] build_cspace vs naive dilation: mismatches=%d  %s"
          % (n_diff, "OK" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    torch.manual_seed(0)
    results = {
        "A (analytic)": check_A(),
        "B (cost_rays_3d vs naive)": check_B(),
        "C (build_cspace vs naive)": check_C(),
    }
    print("\n" + "=" * 60)
    for name, ok in results.items():
        print("  %-28s %s" % (name, "PASS" if ok else "FAIL"))
    all_ok = all(results.values())
    print("=" * 60)
    print("ALL PASS" if all_ok else "SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)
