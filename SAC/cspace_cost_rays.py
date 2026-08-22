"""
PyTorch re-implementation of the `c_space_mapper` "3D cost ray" pipeline
(NavRL-style configuration-space sampling) used by the CNN SAC variant.

This reproduces, in a fully vectorized / GPU-friendly form, the computation done
by the ROS C++ node `c_space_mapper/mapper_node.cpp` + `configuration_mapper.cpp`:

    2D occupancy grid
        -> 3D configuration space  (x, y, theta)   [footprint collision]
        -> 3D cost rays             (beam, theta)  [3D DDA ray cast]

The output `cost_rays` is exactly the `(num_beams, nz)` Float32MultiArray that
`laser_env.cost_rays_callback()` reshapes into the CNN state, i.e. shape
`(N, num_beams=36, nz=10)` here, then `preprocess_cost_rays()` mirrors
`laser_env._preprocess_cost_rays()`.

Defaults are taken from `catkin_ws/src/c_space_mapper/config/mapper_node.yaml`:

    a = 0.698 (rad)  -> nz = 10 theta layers
    g_min = 0.2 (m)  -> c-space spatial resolution
    footprint = 0.44 x 0.38 m rectangle (Pioneer 3-DX)
    cost_ray_max_range = 5.0 m
    cost_ray_num_beams = 36, h_res = 10 deg

Coordinate convention (documented for reproducibility):
  * Robot-centric. The robot sits at the *center* of a `(2R+1) x (2R+1)` grid,
    `R = ceil(max_range / g_min) + 2`, world origin = robot centre.
  * world -> cell:  `i = R + round(wx / g_min)`   (cell-centre aligned)
  * theta cell:     `k = floor(theta / a)`          (matches the C++ DDA, where
    theta starts at 0 = boundary of cell 0)

Fidelity notes (differences vs the C++ that are negligible for training):
  * The C++ rasterizes only the *edges* of the footprint polygon; we use the
    *filled* polygon (more correct for a convex footprint).
  * Footprint masks are built by 3x3 sub-sampling per cell (point-in-polygon),
    which approximates cell/polygon overlap to ~g_min/3.
  * Sub-cell alignment of the robot centre differs by < g_min (~0.2 m), far
    below the 5 m range and the 0.1 m floor of `preprocess_cost_rays`.

Pure-Python + PyTorch (numpy only for the one-time footprint-mask build).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "DEFAULT_CFG",
    "make_footprint_masks",
    "build_cspace",
    "cost_rays_3d",
    "compute_cost_rays",
    "preprocess_cost_rays",
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
                cx = offsets[mi]
                cy = offsets[mj]
                hit = False
                for sx in sub:
                    for sy in sub:
                        if _point_in_polygon(cx + sx, cy + sy, rot):
                            hit = True
                            break
                    if hit:
                        break
                masks[k, mj, mi] = 1.0 if hit else 0.0
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
                 max_range=5.0, g_min=0.2, a=0.698, acc=None):
    """Cast NavRL-style 3D cost rays through the c-space.

    For each of `num_beams` directions and each theta layer k, the ray sweeps
    its orientation linearly from theta=0 to theta_k=k*a over `max_range`, and
    returns the (x,y) distance to the first occupied c-space voxel — i.e.
    "how far can I drive in this direction while rotating to theta_k".

    Args:
        cspace:      (N, nx, ny, nz) bool, from build_cspace.
        robot_cell:  (N, 2) long tensor, the (i, j) cell of the robot centre.
        acc:         marching step (m). Default g_min/2 (~0.1 m), well below the
                     c-space resolution and the 0.1 m floor of preprocessing.

    Returns:
        (N, num_beams, nz) float tensor of distances in metres; `max_range`
        where no collision is found (matches the C++ convention).
    """
    N, nx, ny, nz = cspace.shape
    dev = cspace.device
    acc = g_min / 2.0 if acc is None else acc
    n_steps = int(math.ceil(max_range / acc)) + 2

    # beam directions (unit vectors), (1, B, 1)
    beam_ang = torch.arange(num_beams, device=dev).float() * math.radians(h_res_deg)
    dx = torch.cos(beam_ang).view(1, num_beams, 1)
    dy = torch.sin(beam_ang).view(1, num_beams, 1)

    # theta sweep rate per layer: d_theta_k = theta_k / max_range, (1, 1, nz)
    k_idx = torch.arange(nz, device=dev).float().view(1, 1, nz)
    d_theta = (k_idx * a) / max_range

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
        # theta cell, (1, 1, nz) -> broadcast to (N, B, nz)
        ck = torch.floor(d_theta * t / a).long().clamp(0, nz - 1)

        ci = ci.expand(N, num_beams, nz)
        cj = cj.expand(N, num_beams, nz)
        ck = ck.expand(N, num_beams, nz)

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
        a=cfg["a"],
    )


def preprocess_cost_rays(cost_rays, max_range=5.0):
    """Mirror laser_env._preprocess_cost_rays().

    Clip min distance to 0.1 m, invert so near obstacles -> high activation,
    and add the channel dim.  Input (N, B, nz) -> (N, 1, B, nz).
    """
    cr = torch.clamp_min(cost_rays, 0.1)
    cr = max_range - cr
    return cr.unsqueeze(1)  # (N, 1, B, nz)


# ---------------------------------------------------------------------------
# Self-test (run inside the container: `python cspace_cost_rays.py`)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
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
    print("self-test OK")
