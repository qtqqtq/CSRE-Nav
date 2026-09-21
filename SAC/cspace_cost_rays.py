"""
NavRL-style configuration-space "cost ray" observation used by the CNN SAC
variant (`Sparrow_V2.SparrowCspaceEnv`).  Fully vectorized / GPU-friendly:

    2D occupancy grid (robot-centric, g_min resolution)
        -> 3D configuration space (x, y, theta)   [footprint collision]
        -> 3D cost rays            (beam, theta)  [per-θ radial scan]

The robot is a rigid footprint of *fixed* heading theta_k = k*a per layer.  For
each horizontal beam direction and each heading layer k, the ray slides the
robot centre along the beam and reports the first radius at which the footprint
(heading theta_k) collides — i.e. layer k is one independent distance image of
the 2-D c-space slice at theta_k (the S_stat construction).  The heading does
NOT turn along the ray; the returned value is the horizontal-plane distance the
robot *centre* can reach before its body first touches, in metres.

Per env the output `cost_rays` is `(num_beams, nz)`.  The live observation is
`(N, num_beams×nz + 4)` after `preprocess_cost_rays()` log-inverts the distance
(near obstacle -> high activation; log amplifies close-range detail) and the
kinematic terms are appended.

`DEFAULT_CFG` below is a small reference geometry (Pioneer 3-DX, 0.2 m cells,
5 m range, nz=10) used only by the `--bench` timing harness and the self-test.
The live Sparrow env does NOT use it: it passes its own cfg derived from
`settings.CSPACE_*` — g_min=0.1 m, a=2*pi/CSPACE_NZ≈0.209 rad, nz=CSPACE_NZ=30,
footprint 1.2x0.14 m, max_range=3.0 m.  To change the observation geometry,
edit `settings.CSPACE_*` (single source of truth), not this dict.

Coordinate convention (documented for reproducibility):
  * Robot-centric. The robot sits at the *center* of a `(2R+1) x (2R+1)` grid,
    `R = ceil(max_range / g_min) + 2`, world origin = robot centre.
  * world -> cell:  `i = R + round(wx / g_min)`   (cell-centre aligned)
  * heading layer:  c-space slice k holds the footprint rotated by theta_k = k*a
    (`build_cspace`).  `cost_rays_3d` scans slice k at a *constant* k, so no
    heading cell is stepped along the ray.

Fidelity notes (negligible for training):
  * Footprint masks are built by 3x3 sub-sampling per cell (point-in-polygon),
    which approximates cell/polygon overlap to ~g_min/3.
  * The centre line is stepped at acc = g_min/2 and sampled with `round` to the
    nearest g_min cell centre, so returned distances are quantized to ~g_min.

Pure-Python + PyTorch (numpy only for the one-time footprint-mask build).
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "DEFAULT_CFG",
    "make_footprint_masks",
    "build_cspace",
    "cost_rays_3d",
    "cost_ray_intervals_3d",
    "compute_cost_rays",
    "preprocess_cost_rays",
    "preprocess_cost_ray_intervals",
]

DEFAULT_CFG = {
    "g_min": 0.2,          # c-space spatial resolution (m)
    "a": 0.698,            # theta resolution (rad), ~= 2*pi/9  -> nz = 10
    "nz": 10,              # number of theta layers
    "footprint": [         # robot footprint polygon (m), centred at origin
        [0.22, 0.19], [0.22, -0.19], [-0.22, -0.19], [-0.22, 0.19],
    ],
    "cost_ray_max_range": 5.0,
    "cost_ray_num_beams": 36,
    "cost_ray_h_res_degrees": 10.0,
}


# ---------------------------------------------------------------------------
# Footprint masks (one-time, on CPU with numpy — tiny)
# ---------------------------------------------------------------------------
def _point_in_polygon(x, y, poly):
    """Even-odd ray-casting point-in-polygon test.

    `poly` is a (V, 2) array; `x`, `y` are broadcastable to a common shape.
    Returns a bool array of that shape.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    shape = np.broadcast(x, y).shape
    inside = np.zeros(shape, dtype=bool)
    V = poly.shape[0]
    j = V - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(V):
            xi, yi = poly[i]
            xj, yj = poly[j]
            cond = ((yi > y) != (yj > y)) & (
                x < (xj - xi) * (y - yi) / (yj - yi) + xi
            )
            inside ^= cond
            j = i
    return inside


def make_footprint_masks(g_min, a, nz, footprint, subsample=3):
    """Build the footprint dilation masks for every theta layer.

    Returns a float32 array of shape (nz, m, m) where `mask[k]` is the footprint
    rotated by `theta = k*a`, rasterized onto a g_min-resolution grid.  A cell is
    set to 1 if any of its `subsample x subsample` sub-points falls inside the
    rotated polygon (approximating cell/polygon overlap).

    The mask is used as a dilation structuring element in `build_cspace`.
    """
    footprint = np.asarray(footprint, dtype=np.float64)  # (V, 2)
    max_ext = float(np.abs(footprint).max())             # ~0.22 m
    half = int(math.ceil(max_ext / g_min)) + 1           # cells
    m = 2 * half + 1
    masks = np.zeros((nz, m, m), dtype=np.float32)

    # cell-centre offsets (m), relative to the robot centre
    offsets = (np.arange(m, dtype=np.float64) - half) * g_min
    # sub-point offsets within a cell (m)
    sub = ((np.arange(subsample, dtype=np.float64) + 0.5) / subsample - 0.5) * g_min

    for k in range(nz):
        theta = k * a
        c, s = math.cos(theta), math.sin(theta)
        # rotate vertices by +theta (same convention as footprintCostAtPose)
        rot = footprint @ np.array([[c, s], [-s, c]])   # (V, 2)
        for mi in range(m):
            for mj in range(m):
                cx = offsets[mi]   # +x = forward
                cy = offsets[mj]   # +y = left
                hit = False
                for sx in sub:
                    for sy in sub:
                        if _point_in_polygon(cx + sx, cy + sy, rot):
                            hit = True
                            break
                    if hit:
                        break
                # store as (forward, left) so dim0/dim1 line up with the
                # occupancy grid's (nx=forward, ny=left) in build_cspace.
                masks[k, mi, mj] = 1.0 if hit else 0.0
    return masks


# ---------------------------------------------------------------------------
# 3D configuration space (x, y, theta) via footprint dilation
# ---------------------------------------------------------------------------
def build_cspace(occ_grid, masks):
    """Dilate a 2D occupancy grid by the footprint to get the 3D c-space.

    Args:
        occ_grid: (N, nx, ny) binary tensor (1 = occupied obstacle cell),
                  at `g_min` resolution.
        masks:    (nz, m, m) float32 footprint masks from make_footprint_masks.

    Returns:
        cspace: (N, nx, ny, nz) bool tensor; cspace[n, i, j, k] == True iff the
        footprint (rotated by theta=k*a, centred on cell (i,j)) overlaps an
        obstacle cell — i.e. configuration (i,j,k) is in collision.
    """
    masks_t = torch.as_tensor(masks, dtype=torch.float32)          # (nz, m, m)
    masks_t = masks_t.unsqueeze(1)                                 # (nz, 1, m, m)
    occ = occ_grid.float().unsqueeze(1)                            # (N, 1, nx, ny)
    pad = masks_t.shape[-1] // 2
    out = F.conv2d(occ, masks_t, padding=pad)                      # (N, nz, nx, ny)
    cspace = (out > 0.5)                                           # bool
    return cspace.permute(0, 2, 3, 1)                              # (N, nx, ny, nz)


# ---------------------------------------------------------------------------
# 3D cost ray cast (beam, theta)
# ---------------------------------------------------------------------------
def cost_rays_3d(cspace, robot_cell, num_beams=36, h_res_deg=10.0,
                 max_range=5.0, g_min=0.2, acc=None):
    """Cast NavRL-style cost rays through the c-space (fixed heading per layer).

    Model B semantics: the robot keeps a *constant* heading while its centre
    slides along the beam — (beam b, layer k) is the radial scan of the 2-D
    c-space slice at orientation theta_k = k*a, in horizontal direction b, from
    the robot centre.  Layer k thus answers "if the robot faces theta_k, how far
    can its centre reach along direction b" — one independent distance image per
    orientation layer.  The heading does NOT turn along the ray.

    Args:
        cspace:      (N, nx, ny, nz) bool, from build_cspace.
        robot_cell:  (N, 2) long tensor, the (i, j) cell of the robot centre.
        acc:         marching step (m). Default g_min/2 (~0.1 m), well below the
                     c-space resolution and the 0.1 m floor of preprocessing.

    Returns:
        (N, num_beams, nz) float tensor of distances in metres; `max_range`
        where no collision is found.
    """
    N, nx, ny, nz = cspace.shape
    dev = cspace.device
    acc = g_min / 2.0 if acc is None else acc
    n_steps = int(math.ceil(max_range / acc)) + 2

    # beam directions (unit vectors), (1, B, 1)
    beam_ang = torch.arange(num_beams, device=dev).float() * math.radians(h_res_deg)
    dx = torch.cos(beam_ang).view(1, num_beams, 1)
    dy = torch.sin(beam_ang).view(1, num_beams, 1)

    # constant heading layer per ray: layer k always queries slice k
    k_layer = torch.arange(nz, device=dev).view(1, 1, nz)      # (1,1,nz)

    # robot centre cell, (N, 1, 1)
    ri = robot_cell[:, 0].float().view(N, 1, 1)
    rj = robot_cell[:, 1].float().view(N, 1, 1)

    # flattened c-space for gather-based occupancy lookup
    cspace_flat = cspace.to(torch.uint8).reshape(N, nx * ny * nz)
    stride_ij = ny * nz

    result = torch.full((N, num_beams, nz), max_range, device=dev)
    active = torch.ones((N, num_beams, nz), device=dev, dtype=torch.bool)

    for s in range(1, n_steps + 1):
        t = s * acc
        # world -> cell offset (cell-centre aligned), (N, B, 1)
        ci = (ri + torch.round(dx * t / g_min)).long()
        cj = (rj + torch.round(dy * t / g_min)).long()

        ci = ci.expand(N, num_beams, nz)
        cj = cj.expand(N, num_beams, nz)
        ck = k_layer.expand(N, num_beams, nz)                  # (N, B, nz)

        in_bounds = (ci >= 0) & (ci < nx) & (cj >= 0) & (cj < ny)

        flat = ci * stride_ij + cj * nz + ck                    # (N, B, nz)
        flat_safe = flat.clamp(0, nx * ny * nz - 1)
        occ = cspace_flat.gather(1, flat_safe.reshape(N, -1))
        occ = occ.reshape(N, num_beams, nz)                     # (N, B, nz)

        new_hit = active & in_bounds & (occ > 0)
        result = torch.where(new_hit, torch.full_like(result, t), result)
        active = active & ~new_hit & in_bounds

        if not active.any():
            break

    return result


def cost_ray_intervals_3d(cspace, robot_cell, num_beams=36,
                          h_res_deg=10.0, max_range=5.0,
                          g_min=0.2, acc=None):
    """Return the entry and exit distance of the collision interval per ray.

    The map construction used by Sparrow produces at most one contiguous
    collision interval along each fixed ``(beam, theta)`` ray.  This function
    therefore records both boundaries of that interval instead of stopping at
    the first occupied sample as :func:`cost_rays_3d` does.

    The returned interval is half-open, ``[entry, exit)``.  A clear ray is
    encoded as ``(max_range, max_range)``; an interval continuing past the
    observation window has ``exit == max_range``.  Sampling starts at zero so
    an orientation that already collides at the robot centre has ``entry=0``.

    Args are identical to :func:`cost_rays_3d`.

    Returns:
        ``(N, 2, num_beams, nz)`` in metres. Channel 0 is entry and channel 1
        is exit.
    """
    N, nx, ny, nz = cspace.shape
    dev = cspace.device
    acc = g_min / 2.0 if acc is None else acc
    n_steps = int(math.ceil(max_range / acc))

    beam_ang = torch.arange(num_beams, device=dev).float() \
        * math.radians(h_res_deg)
    dx = torch.cos(beam_ang).view(1, num_beams, 1)
    dy = torch.sin(beam_ang).view(1, num_beams, 1)
    k_layer = torch.arange(nz, device=dev).view(1, 1, nz)

    ri = robot_cell[:, 0].float().view(N, 1, 1)
    rj = robot_cell[:, 1].float().view(N, 1, 1)
    cspace_flat = cspace.to(torch.uint8).reshape(N, nx * ny * nz)
    stride_ij = ny * nz

    entry = torch.full((N, num_beams, nz), max_range, device=dev)
    exit_distance = torch.full_like(entry, max_range)
    # phase: 0 = looking for entry, 1 = looking for exit, 2 = complete
    phase = torch.zeros((N, num_beams, nz), dtype=torch.uint8, device=dev)

    # max_range is a sentinel/boundary rather than a collision sample.  This
    # makes (R,R) unambiguously mean no interval within the local window.
    for s in range(n_steps):
        t = s * acc
        if t >= max_range:
            break

        ci = (ri + torch.round(dx * t / g_min)).long() \
            .expand(N, num_beams, nz)
        cj = (rj + torch.round(dy * t / g_min)).long() \
            .expand(N, num_beams, nz)
        ck = k_layer.expand(N, num_beams, nz)

        in_bounds = (ci >= 0) & (ci < nx) & (cj >= 0) & (cj < ny)
        flat = ci * stride_ij + cj * nz + ck
        flat_safe = flat.clamp(0, nx * ny * nz - 1)
        occupied = cspace_flat.gather(1, flat_safe.reshape(N, -1))
        occupied = occupied.reshape(N, num_beams, nz).bool() & in_bounds

        new_entry = (phase == 0) & occupied
        entry = torch.where(new_entry, torch.full_like(entry, t), entry)
        phase = torch.where(new_entry, torch.ones_like(phase), phase)

        new_exit = (phase == 1) & ~occupied
        exit_distance = torch.where(
            new_exit, torch.full_like(exit_distance, t), exit_distance
        )
        phase = torch.where(new_exit, torch.full_like(phase, 2), phase)

        if torch.all(phase == 2):
            break

    return torch.stack((entry, exit_distance), dim=1)


# ---------------------------------------------------------------------------
# Top-level convenience + preprocessing (mirrors laser_env)
# ---------------------------------------------------------------------------
def compute_cost_rays(occ_grid, robot_cell, cfg=None):
    """End-to-end: occ_grid -> 3D c-space -> cost rays.

    Args:
        occ_grid:   (N, nx, ny) binary tensor at `g_min` resolution.
        robot_cell: (N, 2) long tensor of the robot's (i, j) cell.
        cfg:        optional overrides of DEFAULT_CFG.

    Returns:
        (N, num_beams, nz) float distances in metres.
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    masks = make_footprint_masks(cfg["g_min"], cfg["a"], cfg["nz"],
                                 cfg["footprint"])
    cspace = build_cspace(occ_grid, masks)
    return cost_rays_3d(
        cspace, robot_cell,
        num_beams=cfg["cost_ray_num_beams"],
        h_res_deg=cfg["cost_ray_h_res_degrees"],
        max_range=cfg["cost_ray_max_range"],
        g_min=cfg["g_min"],
    )


# Min representable ray distance (m) used as the log floor.  The live Sparrow
# c-space grid is g_min = 0.1 m and ray distances are quantized to ~g_min, so
# nothing meaningful exists below one cell (a smaller floor only adds empty
# bins); at/under this value the ray is a footprint contact.
LOG_CR_MIN_DIST = 0.1


def preprocess_cost_rays(cost_rays, max_range=5.0):
    """Near-field-sensitive log encoding of the cost-ray distances (1 channel).

    Mirror laser_env._preprocess_cost_rays().  d (m) is clipped to
    [LOG_CR_MIN_DIST, max_range] then encoded as

        p = ln(max_range / d)            (near obstacle -> high activation)

    which replaces the old *linear* inversion (max_range - d).  Because
    |dp/dd| = 1/d, close-range (sub-metre) differences get amplified — the
    regime the robot must react to — at the cost of flattening far rays
    (d ≳ 2 m).  Floor = LOG_CR_MIN_DIST (≈ c-space grid cell), where any
    smaller distance is a footprint contact; upper clip keeps ln(·) >= 0.
    Input (N, B, nz) -> (N, 1, B, nz).
    """
    cr = cost_rays.clamp(LOG_CR_MIN_DIST, max_range)
    cr = torch.log(max_range / cr)
    return cr.unsqueeze(1)  # (N, 1, B, nz)


def preprocess_cost_ray_intervals(intervals, max_range=5.0, epsilon=0.05):
    """Invertibly encode interval rays as entry proximity and radial width.

    Input and output both have shape ``(N, 2, B, nz)``.  Raw channels are
    ``[entry, exit]`` in metres; processed channels are:

    ``entry_proximity = 1 - log1p(entry/epsilon) / log1p(R/epsilon)``
    ``collision_width = (exit-entry) / R``

    Thus near collisions have high activation, while the second channel tells
    the policy how much radial space the collision region occupies.  A clear
    ray ``(R,R)`` maps exactly to ``(0,0)``.  The mapping is invertible over the
    sampled interval distances.
    """
    if intervals.ndim != 4 or intervals.shape[1] != 2:
        raise ValueError(
            "intervals must have shape (N,2,B,nz), got %s" %
            (tuple(intervals.shape),)
        )
    if max_range <= 0 or epsilon <= 0:
        raise ValueError("max_range and epsilon must be positive")

    entry = intervals[:, 0].clamp(0.0, max_range)
    exit_distance = intervals[:, 1].clamp(0.0, max_range)
    if torch.any(exit_distance < entry):
        raise ValueError("collision interval exit must be >= entry")

    denom = math.log1p(max_range / epsilon)
    entry_proximity = 1.0 - torch.log1p(entry / epsilon) / denom
    width = (exit_distance - entry) / max_range
    return torch.stack((entry_proximity, width), dim=1)


# ---------------------------------------------------------------------------
# Computation-time test (`python cspace_cost_rays.py --bench`)
# ---------------------------------------------------------------------------
def _measure(fn, *args, repeat=20, warmup=3, sync=False, **kwargs):
    """Wall-clock seconds per call of `fn(*args, **kwargs)`.

    Returns `(min, mean, std)` over `repeat` timed calls after `warmup` runs.
    The minimum is reported downstream — it is the best proxy for steady-state
    latency (one outlier interrupt otherwise skews the average).  With
    `sync=True` a `torch.cuda.synchronize()` flushes the launch queue before
    and after each timed call so GPU kernels are fully counted.
    """
    if sync:
        torch.cuda.synchronize()
    for _ in range(warmup):
        fn(*args, **kwargs)
        if sync:
            torch.cuda.synchronize()
    ts = []
    for _ in range(repeat):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        if sync:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[0], float(np.mean(ts)), float(np.std(ts))


def _bench_main(argv=None):
    """Time each pipeline stage; the harness behind `python cspace_cost_rays.py --bench`.

    Geometry defaults to the Sparrow env (`Sparrow_V2.py`): footprint
    ``settings.ROBOT_FOOTPRINT`` = 1.2 x 0.14 m, g_min = 0.2 m -> R = 27 ->
    55 x 55 grid, max_range = 5.0 m, 36 beams, 10 theta layers.  The occupancy
    map is *empty* so every ray sweeps the full `max_range` (no early exit) —
    i.e. the reported per-call time is the worst-case / steady-state cost.
    """
    import argparse

    p = argparse.ArgumentParser(
        description="Time each stage of the c-space cost-ray pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--batch", default="1,32,256",
                   help="comma-separated env-batch sizes N to sweep")
    p.add_argument("--repeat", type=int, default=20,
                   help="timed calls per measurement (min is reported)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--device", default="auto",
                   help="cpu | cuda | cuda:N | auto (first available device)")
    p.add_argument("--footprint", default="sparrow",
                   help="sparrow (1.2x0.14 m) | pioneer (DEFAULT_CFG) | 'L,W' m")
    args = p.parse_args(argv)

    cfg = dict(DEFAULT_CFG)
    fp_arg = args.footprint.lower()
    if fp_arg == "sparrow":                                  # settings.ROBOT_FOOTPRINT
        fp = [[0.6, 0.07], [0.6, -0.07], [-0.6, -0.07], [-0.6, 0.07]]
    elif fp_arg == "pioneer":
        fp = DEFAULT_CFG["footprint"]
    else:
        L, W = [float(x) for x in fp_arg.split(",")]
        hl, hw = L / 2.0, W / 2.0
        fp = [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]]
    cfg["footprint"] = fp

    batch_sizes = [int(x) for x in args.batch.split(",")]
    R = int(math.ceil(cfg["cost_ray_max_range"] / cfg["g_min"])) + 2
    G = 2 * R + 1

    if torch.cuda.is_available():
        devices = ["cuda:0"] if args.device == "auto" else [args.device]
    else:
        devices = ["cpu"] if args.device == "auto" else [args.device]

    # make_footprint_masks is pure numpy/CPU and runs once at env init; build the
    # masks a single time and move the float32 tensor to each device under test.
    masks_cpu = make_footprint_masks(cfg["g_min"], cfg["a"], cfg["nz"], fp)  # (nz,m,m)
    m_min, m_mean, _ = _measure(make_footprint_masks, cfg["g_min"], cfg["a"],
                                cfg["nz"], fp, repeat=min(args.repeat, 5), warmup=1)

    print(f"cspace cost-ray timing (empty map -> full {cfg['cost_ray_max_range']} m sweep, "
          f"grid {G}x{G}x{cfg['nz']}, footprint {fp[0]})")
    print(f"  make_footprint_masks (one-time, CPU): min={m_min*1e3:.3f} ms  "
          f"mean={m_mean*1e3:.3f} ms")

    for dev_s in devices:
        dev = torch.device(dev_s)
        sync = dev.type == "cuda"
        masks_t = torch.as_tensor(masks_cpu, device=dev)  # (nz, m, m)
        print(f"\n  device: {dev_s}   min over {args.repeat} calls after {args.warmup} warmups")
        print(f"    {'N':>5} {'cspace':>10} {'cost_rays3d':>13} {'preproc':>10} "
              f"{'obs/call':>12} {'obs/env':>11} {'steps/s':>10}")
        for N in batch_sizes:
            # empty occupancy -> every beam sweeps the full range (worst case)
            occ = torch.zeros(N, G, G, dtype=torch.bool, device=dev)
            robot_cell = torch.full((N, 2), R, dtype=torch.long, device=dev)
            cr_kw = dict(num_beams=cfg["cost_ray_num_beams"],
                         h_res_deg=cfg["cost_ray_h_res_degrees"],
                         max_range=cfg["cost_ray_max_range"],
                         g_min=cfg["g_min"])

            b = _measure(build_cspace, occ, masks_t, repeat=args.repeat,
                         warmup=args.warmup, sync=sync)[0] * 1e3
            cspace = build_cspace(occ, masks_t)
            r = _measure(cost_rays_3d, cspace, robot_cell, repeat=args.repeat,
                         warmup=args.warmup, sync=sync, **cr_kw)[0] * 1e3
            cr = cost_rays_3d(cspace, robot_cell, **cr_kw)
            pp = _measure(preprocess_cost_rays, cr, max_range=cfg["cost_ray_max_range"],
                          repeat=args.repeat, warmup=args.warmup, sync=sync)[0] * 1e3
            tot = b + r + pp
            print(f"    {N:>5} {b:>10.3f} {r:>13.3f} {pp:>10.3f} "
                  f"{tot:>12.3f} {tot / N:>11.3f} {N / (tot * 1e-3):>10.1f}")


# ---------------------------------------------------------------------------
# Self-test (run inside the container: `python cspace_cost_rays.py`; add
# `--bench` for the computation-time test above)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if "--bench" in sys.argv:
        # hand the sub-harness only the args *after* the trigger flag
        raise SystemExit(_bench_main(sys.argv[sys.argv.index("--bench") + 1:]))

    torch.manual_seed(0)
    g_min, max_range = DEFAULT_CFG["g_min"], DEFAULT_CFG["cost_ray_max_range"]
    R = int(math.ceil(max_range / g_min)) + 2
    G = 2 * R + 1
    N = 1
    robot_cell = torch.tensor([[R, R]], dtype=torch.long)

    # A vertical wall of obstacle cells at x = R + 10 (2.0 m to the right).
    occ = torch.zeros(N, G, G, dtype=torch.bool)
    occ[0, R + 10, :] = True

    cr = compute_cost_rays(occ, robot_cell)
    print("cost_rays shape:", tuple(cr.shape))
    print("beam 0   (+x, toward wall):", cr[0, 0].tolist())
    print("beam 18  (-x, away):        ", cr[0, 18].tolist())
    print("beam 9   (+y, side):        ", cr[0, 9].tolist())

    # sanity assertions
    assert tuple(cr.shape) == (N, DEFAULT_CFG["cost_ray_num_beams"], DEFAULT_CFG["nz"])
    assert torch.all(cr[0, 0] < 2.0), "beam 0 should hit the 2 m wall"
    assert torch.all(cr[0, 18] == max_range), "beam 18 should see no obstacle"

    # Exact interval-ray test on a directly constructed one-theta c-space.
    # Along beam 0 (+x), cells +3..+5 are occupied: [0.6 m, 1.2 m).
    interval_cs = torch.zeros(N, G, G, 1, dtype=torch.bool)
    interval_cs[0, R + 3:R + 6, R, 0] = True
    intervals = cost_ray_intervals_3d(
        interval_cs, robot_cell, num_beams=4, h_res_deg=90.0,
        max_range=max_range, g_min=g_min, acc=g_min,
    )
    assert tuple(intervals.shape) == (N, 2, 4, 1)
    assert torch.isclose(intervals[0, 0, 0, 0], torch.tensor(0.6))
    assert torch.isclose(intervals[0, 1, 0, 0], torch.tensor(1.2))
    assert torch.all(intervals[0, :, 2, 0] == max_range)

    proc = preprocess_cost_ray_intervals(
        intervals, max_range=max_range, epsilon=g_min / 2.0
    )
    denom = math.log1p(max_range / (g_min / 2.0))
    entry_recovered = (g_min / 2.0) * (
        torch.exp((1.0 - proc[:, 0]) * denom) - 1.0
    )
    exit_recovered = entry_recovered + proc[:, 1] * max_range
    assert torch.allclose(entry_recovered, intervals[:, 0], atol=1e-5)
    assert torch.allclose(exit_recovered, intervals[:, 1], atol=1e-5)

    radial = torch.arange(0.0, max_range, g_min)
    reconstructed = (
        (radial >= intervals[0, 0, 0, 0])
        & (radial < intervals[0, 1, 0, 0])
    )
    expected = (radial >= 0.6) & (radial < 1.2)
    assert torch.equal(reconstructed, expected)
    print("self-test OK")
