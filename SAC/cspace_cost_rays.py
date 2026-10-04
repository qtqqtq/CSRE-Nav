"""
NavRL-style configuration-space "cost ray" observation used by the CNN SAC
variant (`Sparrow_V2.SparrowCspaceEnv`).  Fully vectorized / GPU-friendly:

    2D occupancy grid (robot-centric, g_min resolution)
        -> 3D configuration space (x, y, theta)   [footprint collision]
        -> 3D cost rays            (beam, theta)  [per-θ radial scan]

The robot is a rigid footprint of *fixed* heading theta_k = k*a per layer.  For
each horizontal beam direction and heading layer k, the ray slides the robot
centre through the 2-D c-space slice at theta_k.  The live observation records
a configurable number of consecutive radial distances: robot-to-first-obstacle,
then alternating occupied and free widths.  The heading does NOT turn along
the ray.

Per env the live ray tensor is `(num_intervals, num_beams, nz)`, normalized by
the maximum range, and four kinematic terms are appended after flattening.

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

Pure-Python + PyTorch (NumPy builds fixed or per-episode footprint masks).
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
    "make_rectangle_union_masks",
    "build_cspace",
    "cost_rays_3d",
    "cost_ray_intervals_3d",
    "compute_cost_rays",
    "preprocess_cost_rays",
    "preprocess_cost_ray_intervals",
    # analytic point-set backend (no grid)
    "rectangles_to_polygons",
    "polygon_constraints",
    "footprint_constraints",
    "footprint_constraint_batch",
    "pointset_intervals",
    "pointset_batch",
    "pointset_collision",
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
    "cost_ray_num_intervals": 3,
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


def make_rectangle_union_masks(g_min, a, nz, rectangles, half, subsample=3):
    """Rasterize the union of centered rectangles for each heading layer.

    ``rectangles`` contains (length_m, width_m, angle_rad) rows in the robot
    frame. The fixed ``half`` keeps all episode masks the same (nz, m, m) shape.
    """
    offsets = (np.arange(-half, half + 1, dtype=np.float64) * g_min)
    sub = ((np.arange(subsample) + 0.5) / subsample - 0.5) * g_min
    x = offsets[:, None, None, None] + sub[None, None, :, None]
    y = offsets[None, :, None, None] + sub[None, None, None, :]
    masks = np.zeros((nz, 2 * half + 1, 2 * half + 1), dtype=np.float32)
    for k in range(nz):
        covered = np.zeros_like(x + y, dtype=bool)
        for length, width, angle in rectangles:
            c, s = math.cos(angle + k * a), math.sin(angle + k * a)
            local_x = x * c + y * s
            local_y = -x * s + y * c
            covered |= ((abs(local_x) <= length / 2) &
                        (abs(local_y) <= width / 2))
        masks[k] = covered.any(axis=(2, 3))
    return masks


# ---------------------------------------------------------------------------
# 3D configuration space (x, y, theta) via footprint dilation
# ---------------------------------------------------------------------------
def build_cspace(occ_grid, masks, known_free_grid=None):
    """Build collision configurations from occupancy, footprint, and known space.

    Args:
        occ_grid: (N, nx, ny) bool tensor (1 = occupied obstacle cell).
        masks:    (nz, m, m) shared or (N, nz, m, m) per-env masks.
        known_free_grid: optional (N, nx, ny) bool tensor (1 = observed free).
                  When provided, a pose is valid if its footprint overlaps at least one known
                  free cell and overlaps no occupied cell. Thus a footprint
                  entirely inside unknown space is collision; partial unknown
                  coverage is accepted when known free space is also covered.

    Returns:
        cspace: (N, nx, ny, nz) bool tensor; cspace[n, i, j, k] == True iff the
        footprint overlaps an obstacle, or (when known_free_grid is provided)
        contains no known-free cells. Unknown cells do not count as occupied.
    """
    masks_t = torch.as_tensor(masks, dtype=torch.float32,
                              device=occ_grid.device)
    per_env = masks_t.ndim == 4
    if per_env:
        n, nz, m, _ = masks_t.shape
        if n != occ_grid.shape[0]:
            raise ValueError("per-env footprint batch must match occupancy batch")
        masks_t = masks_t.reshape(n * nz, 1, m, m)
        occ = occ_grid.float().unsqueeze(0)                      # (1, N, nx, ny)
        groups = n
    else:
        masks_t = masks_t.unsqueeze(1)                           # (nz, 1, m, m)
        occ = occ_grid.float().unsqueeze(1)                      # (N, 1, nx, ny)
        groups = 1
    pad = masks_t.shape[-1] // 2
    out = F.conv2d(occ, masks_t, padding=pad, groups=groups)
    if per_env:
        out = out.reshape(n, nz, *out.shape[-2:])
    cspace = out > 0.5                                             # occupied overlap
    if known_free_grid is not None:
        known_free = known_free_grid.float().unsqueeze(0 if per_env else 1)
        known_count = F.conv2d(known_free, masks_t, padding=pad,
                               groups=groups)
        if per_env:
            known_count = known_count.reshape(n, nz, *known_count.shape[-2:])
        entirely_unknown = known_count <= 0.5
        cspace = cspace | entirely_unknown
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
                          g_min=0.2, acc=None, num_intervals=3):
    """Return consecutive radial interval lengths along each c-space ray.

    The channels start with ``first_entry`` and then alternate between occupied
    and free widths::

        [first_entry, first_occupied_width, first_free_width,
         second_occupied_width, second_free_width, ...]

    ``num_intervals`` controls the number of returned channels.  A clear ray is
    encoded as ``(max_range, 0, ...)``.  Once a transition has been observed,
    the currently open interval extends to ``max_range`` if its closing
    transition is not observed.  Sampling starts at zero, so a configuration
    already in collision has ``first_entry == 0``.

    Other args are identical to :func:`cost_rays_3d`.

    Returns:
        ``(N, num_intervals, num_beams, nz)`` in metres.
    """
    if isinstance(num_intervals, bool) or not isinstance(num_intervals, int):
        raise TypeError("num_intervals must be an integer")
    if num_intervals < 1:
        raise ValueError("num_intervals must be at least 1")

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

    boundaries = torch.full(
        (N, num_intervals, num_beams, nz), max_range, device=dev
    )
    # phase is the number of observed free/occupied transitions.  Even phases
    # wait for an occupied sample; odd phases wait for a free sample.
    phase = torch.zeros((N, num_beams, nz), dtype=torch.int32, device=dev)

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

        active = phase < num_intervals
        expected_occupied = (phase % 2) == 0
        transition = active & (occupied == expected_occupied)
        for interval_idx in range(num_intervals):
            found = transition & (phase == interval_idx)
            boundaries[:, interval_idx] = torch.where(
                found,
                torch.full_like(boundaries[:, interval_idx], t),
                boundaries[:, interval_idx],
            )
        phase = phase + transition.to(phase.dtype)

        if torch.all(phase == num_intervals):
            break

    intervals = torch.zeros_like(boundaries)
    intervals[:, 0] = boundaries[:, 0]
    for interval_idx in range(1, num_intervals):
        previous_boundary_seen = phase >= interval_idx
        intervals[:, interval_idx] = torch.where(
            previous_boundary_seen,
            boundaries[:, interval_idx] - boundaries[:, interval_idx - 1],
            intervals[:, interval_idx],
        )
    return intervals


# ---------------------------------------------------------------------------
# Top-level convenience + preprocessing
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

    Distance d (m) is clipped to
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
    """Normalize ray-segment distance channels to ``[0, 1]``.

    Input ``(N, C, B, nz)`` is in metres, where ``C`` is the configured number
    of consecutive radial intervals.  Dividing every distance by ``max_range``
    gives the CNN a common numeric scale.  ``epsilon`` remains an accepted
    argument for compatibility with existing callers but is no longer used.
    """
    if intervals.ndim != 4 or intervals.shape[1] < 1:
        raise ValueError(
            "intervals must have shape (N,C,B,nz) with C >= 1, got %s" %
            (tuple(intervals.shape),)
        )
    if max_range <= 0:
        raise ValueError("max_range must be positive")
    if torch.any(intervals < 0):
        raise ValueError("ray-segment distances must be non-negative")
    return intervals.clamp(0.0, max_range) / max_range


# ---------------------------------------------------------------------------
# Analytic point-set backend (no grid): obstacle = lidar hit points
# ---------------------------------------------------------------------------
# The grid backend rasterizes the footprint (`make_footprint_masks`) and dilates
# the occupancy grid, so surfaces land on `g_min` cell centres and two adjacent
# hit cells can each round the *wrong* way, opening a phantom gap in a wall that
# should be closed.  This backend drops the grid: an obstacle is a point, and
# "would driving along beam b at heading theta_k touch that point?" is answered
# in closed form.
#
# The robot drives along `u_b` with its heading FIXED at theta_k and the
# obstacle points do not move, so at travel distance t the point sits at
# `p - t*u_b` in the robot frame.  Contact happens while that sits inside the
# rotated footprint F.  With `n_j` the outward edge normals of a convex piece
# and `h_j = <v, n_j>` its edge offsets, `x in R(theta) F` is
# `<R(theta)^-1 x, n_j> <= h_j` for every j, which is one linear inequality per
# edge in the travel distance:
#
#     t * <u_b, R(theta_k) n_j>  >=  <p, R(theta_k) n_j> - h_j
#
# Reading it per edge: the left factor is how fast the point closes on that
# edge while driving (a dimensionless cosine), the right side is how far
# outside that edge it started (metres).  A positive rate gives a lower bound
# on t, a negative rate an upper bound, and a rate of ~0 means the ray runs
# parallel to that edge, leaving only lateral feasibility (<p, n_j> <= h_j).
# Edges are ANDed (intersect the half-planes) while pieces, points and theta
# layers are ORed (take the minimum) -- unions, not intersections.
#
# This is an *alternative* to the grid path, not a refinement of it: it shares
# the c-space observation's shapes and semantics but none of its quantization.
def rectangles_to_polygons(rectangles):
    """``(length_m, width_m, angle_rad)`` rows -> convex (V, 2) polygons.

    Reproduces the shape `make_rectangle_union_masks` rasterizes, i.e. the
    axis-aligned box ``|x| <= L/2, |y| <= W/2`` rotated by ``+angle`` in the
    robot frame, so the analytic and raster backends describe the same
    footprint for the same rectangle list.
    """
    polygons = []
    for length, width, angle in rectangles:
        hl, hw = length / 2.0, width / 2.0
        c, s = math.cos(angle), math.sin(angle)
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        box = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]],
                       dtype=np.float64)
        polygons.append(box @ rot.T)          # row i -> rot @ box[i]
    return polygons


def polygon_constraints(polygons):
    """Convex polygons -> per-piece outward edge normals and offsets.

    Returns two lists of ``(V, 2)`` / ``(V,)`` arrays with
    ``h_j = <v, n_j>`` for either vertex of edge j (signed metres; negative
    when the edge lies on the far side of the robot centre).  The normal's sign
    is fixed numerically by the centroid test rather than assumed from the
    winding order: convex polygons always contain their vertex centroid, so
    ``<centroid, n_j> > h_j`` means ``n_j`` points inward and must be flipped.
    """
    normals, offsets = [], []
    for polygon in polygons:
        polygon = np.asarray(polygon, dtype=np.float64)
        centroid = polygon.mean(axis=0)
        edge = np.roll(polygon, -1, axis=0) - polygon
        normal = np.stack([edge[:, 1], -edge[:, 0]], axis=1)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        offset = (polygon * normal).sum(axis=1)
        flip = (centroid @ normal.T) > offset
        normal[flip] *= -1.0
        offset[flip] *= -1.0
        normals.append(normal)
        offsets.append(offset)
    return normals, offsets


def _edge_cosines(nrm, num_beams, nz, a, h_res_degrees=None, device=None,
                  dtype=None):
    """``c[n,p,b,k,j] = <u_b, R(theta_k) n_j>``, shape ``(N,P,B,nz,V)``.

    Input ``nrm`` is ``(N,P,V,2)``.  Dimensionless: it is how fast the obstacle
    point closes on edge j of piece p while the robot drives along beam b with
    its heading fixed at ``theta_k = k*a``.  ``theta`` only ever rotates the
    footprint (the point set is rotated per step instead); the edge offsets `h`
    are rotation-invariant, so they need no layer axis.
    """
    dtype = nrm.dtype if dtype is None else dtype
    device = nrm.device if device is None else device
    if h_res_degrees is None:
        h_res_degrees = 360.0 / num_beams
    beam_ang = torch.arange(num_beams, device=device, dtype=dtype) \
        * math.radians(h_res_degrees)
    u = torch.stack((torch.cos(beam_ang), torch.sin(beam_ang)), dim=-1)  # (B,2)
    theta = torch.arange(nz, device=device, dtype=dtype) * a
    ct, st = torch.cos(theta), torch.sin(theta)                          # (nz,)
    nx, ny = nrm[..., 0], nrm[..., 1]                                    # (N,P,V)
    fx, fy = nx[None], ny[None]                                          # (1,N,P,V)
    ct, st = ct[:, None, None, None], st[:, None, None, None]            # (nz,1,1,1)
    # R(theta_k) n_j = (c*nx - s*ny, s*nx + c*ny) -> (nz, N, P, V, 2)
    rot = torch.stack((ct * fx - st * fy, st * fx + ct * fy), dim=-1)
    return torch.einsum("bx,knpvx->npbkv", u, rot)


def footprint_constraints(polygons, num_beams, nz, a, h_res_degrees=None,
                          device=None, dtype=torch.float32):
    """Precompute the per-episode edge table for `pointset_intervals`.

    Returns ``nrm (P, V, 2)``, ``h (P, V)`` and ``c (P, B, nz, V)``.
    ``h_res_degrees`` must be the same beam spacing the grid backend scans with
    (`cost_ray_intervals_3d`'s `h_res_deg`, i.e. 360/num_beams), otherwise the
    two backends do not describe the same beams.
    """
    nrm_list, h_list = polygon_constraints(polygons)
    nrm = torch.as_tensor(np.stack(nrm_list), dtype=dtype, device=device)
    h = torch.as_tensor(np.stack(h_list), dtype=dtype, device=device)
    c = _edge_cosines(nrm.unsqueeze(0), num_beams, nz, a, h_res_degrees,
                      device, dtype).squeeze(0)
    return nrm, h, c


def footprint_constraint_batch(polygons_per_env, num_beams, nz, a,
                               h_res_degrees=None, device=None,
                               dtype=torch.float32):
    """Per-environment edge tables, padded to the widest footprint.

    Environment i draws its own footprint (random footprints change per
    episode), so the tables carry a leading env axis: ``nrm (N,P,V,2)``,
    ``h (N,P,V)``, ``c (N,P,B,nz,V)``.  Missing pieces are padded with a zero
    normal and ``h = -1``: the only constraint they add has ``c = 0`` and
    ``g = -h = 1 > 0``, which is infeasible -- exactly what "this piece does not
    exist" should contribute (no contact, no collision).
    """
    tables = [polygon_constraints(polygons) for polygons in polygons_per_env]
    n_env = len(tables)
    p_max = max(len(polygons) for polygons in polygons_per_env)
    v_max = max(len(polygon) for polygons in polygons_per_env
                for polygon in polygons)
    nrm = torch.zeros((n_env, p_max, v_max, 2), dtype=dtype, device=device)
    h = -torch.ones((n_env, p_max, v_max), dtype=dtype, device=device)
    for i, (normals, offsets) in enumerate(tables):
        pieces = len(normals)
        edges = normals[0].shape[0]
        nrm[i, :pieces, :edges] = torch.as_tensor(np.stack(normals),
                                                  dtype=dtype, device=device)
        h[i, :pieces, :edges] = torch.as_tensor(np.stack(offsets), dtype=dtype,
                                                device=device)
    c = _edge_cosines(nrm, num_beams, nz, a, h_res_degrees, device, dtype)
    return nrm, h, c


def _as_env_tables(nrm, h, c, n_env):
    """Accept shared ``(P,V,·)`` tables by broadcasting them over the env axis.

    A single fixed footprint is shared by every environment, so the caller need
    not replicate it; `expand` only makes the leading axis a view.
    """
    if nrm.dim() == 3:
        nrm = nrm.unsqueeze(0)
        h = h.unsqueeze(0)
        if c is not None:
            c = c.unsqueeze(0)
    if nrm.shape[0] == 1 and n_env > 1:
        nrm = nrm.expand(n_env, *nrm.shape[1:])
        h = h.expand(n_env, *h.shape[1:])
        if c is not None:
            c = c.expand(n_env, *c.shape[1:])
    return nrm, h, c


def pointset_intervals(pts, nrm, h, c, a, c_eps=1e-9):
    """Raw first-contact intervals for a lidar point set.

    ``pts`` is ``(N, M, 2)`` obstacle points in the robot frame, metres.
    Returns ``(entry, exit)``, both ``(N, B, nz)`` and unclipped: ``+inf`` means
    that beam never touches anything, and ``entry == 0`` means the current pose
    is already in contact.  Callers clip to the observation range.

    ``c_eps`` classifies the parallel case.  ``c`` is a cosine in [-1, 1], so
    the threshold is absolute and dimensionless -- never compare `c` with 0
    directly (an axis-aligned beam can give an exact 0 or a 6e-17 residue
    depending on how the angle was computed, and both must take the same
    branch), and never apply a threshold to the metre-valued ``g``.
    """
    N, M, _ = pts.shape
    nrm, h, c = _as_env_tables(nrm, h, c, N)
    P, B, nz, V = c.shape[1], c.shape[2], c.shape[3], c.shape[4]
    dt = pts.dtype
    dev = pts.device
    finfo = torch.finfo(dt)

    # q[n, m, k] = R(-theta_k) p  -- theta only *rotates* the point, t only
    # translates it, and the two never mix in the same expression.
    theta = torch.arange(nz, device=dev, dtype=dt) * a
    ct, st = torch.cos(theta), torch.sin(theta)
    px, py = pts[:, :, 0, None], pts[:, :, 1, None]        # (N, M, 1)
    qx = px * ct + py * st                                 # (N, M, nz)
    qy = -px * st + py * ct

    best_entry = torch.full((N, B, nz), finfo.max, dtype=dt, device=dev)
    best_exit = torch.full_like(best_entry, finfo.max)
    for i in range(P):
        # Edges of one piece AND together, so start from the widest interval.
        t_lo = torch.full((N, M, B, nz), finfo.min, dtype=dt, device=dev)
        t_hi = torch.full_like(t_lo, finfo.max)
        feasible = torch.ones((N, M, B, nz), dtype=torch.bool, device=dev)
        for j in range(V):
            # Per-env edge j of piece i: (N,) normal components, (N,) offset.
            s = (qx * nrm[:, i, j, 0][:, None, None]
                 + qy * nrm[:, i, j, 1][:, None, None])        # (N, M, nz)
            g = s - h[:, i, j][:, None, None]                  # (N, M, nz), metres
            cc = c[:, i, :, :, j][:, None, :, :]               # (N, 1, B, nz)
            # c*t >= g.  c > 0 divides without flipping, giving a lower bound on
            # t (driving forward pushes the point *into* this half-plane, i.e. it
            # enters through this edge); c < 0 flips the sense and bounds t from
            # above (the point leaves again by this edge).
            pos = cc > c_eps                               # t >= g/c  (enters)
            neg = cc < -c_eps                              # t <= g/c  (exits)
            safe = torch.where(pos | neg, cc, torch.ones_like(cc))
            ratio = g[:, :, None, :] / safe
            t_lo = torch.maximum(t_lo, torch.where(pos, ratio, t_lo))
            t_hi = torch.minimum(t_hi, torch.where(neg, ratio, t_hi))
            # Parallel to this edge: no bound on t, only "stay on the inside".
            feasible &= ~((~pos) & (~neg) & (g[:, :, None, :] > 0.0))
        # The interval only counts if it reaches t >= 0: a point *behind* the
        # robot on a forward beam gives t_hi < 0 (it was passed already), which
        # is "no contact", not "contact at 0".  Taking t_lo == -inf as 0 would
        # read as an immediate collision.
        start = t_lo.clamp_min(0.0)
        empty = ~feasible | (t_hi < start)
        entry_i = torch.where(empty, best_entry.new_full((), finfo.max), start)
        exit_i = torch.where(empty, best_entry.new_full((), finfo.max), t_hi)
        # Points are OR too: the nearest one owns the cell.  Its exit comes along
        # as the payload so channel 1 describes *that* contact, not some other
        # point's.  Ties keep the lowest point index -- for one footprint the two
        # blobs have the same shape, so tied entries differ in exit by ~the
        # sampling spacing (centimetres at most).
        entry_i, winner = entry_i.min(dim=1)                    # (N, B, nz)
        exit_i = exit_i.gather(1, winner.unsqueeze(1)).squeeze(1)
        # Min over pieces; ties keep the earlier piece.
        better = entry_i < best_entry
        best_entry = torch.where(better, entry_i, best_entry)
        best_exit = torch.where(better, exit_i, best_exit)
    return best_entry, best_exit


def pointset_batch(pts, nrm, h, c, a, max_range):
    """Observation-ready point-set rays: entry, contact width, and collision.

    ``entry (N, B, nz)`` is the usual first-obstacle distance clipped to
    ``[0, max_range]`` (``max_range`` = nothing in reach).  ``width (N, B, nz)``
    is how far the robot would travel inside that first contact before coming
    out the far side -- the point-set analogue of the grid backend's
    ``first_occupied_width``, which is what channel 1 of the live observation
    holds.  ``collide (N,)`` is the current-pose test at theta = 0.
    """
    entry_raw, exit_ = pointset_intervals(pts, nrm, h, c, a)
    # Clip *after* deciding hit/no-hit: clamped entry == max_range is also the
    # "nothing in reach" value, and exit_ is +inf there.
    hit = entry_raw <= max_range
    entry = entry_raw.clamp(0.0, max_range)
    width = torch.where(hit, (exit_ - entry_raw).clamp_min(0.0),
                        torch.zeros_like(entry))
    return entry, width, pointset_collision(pts, nrm, h)


def pointset_collision(pts, nrm, h):
    """Current-pose collision test: is any point inside any footprint piece?

    This is the ``t = 0`` case of the same inequality set, so it is identically
    ``pointset_batch(...)[0][:, :, 0].min(dim=1).values == 0`` -- one geometry,
    two entry points, asserted equal in the self-test.
    """
    N = pts.shape[0]
    nrm, h, _ = _as_env_tables(nrm, h, None, N)
    inside_any = torch.zeros(N, dtype=torch.bool, device=pts.device)
    for i in range(nrm.shape[1]):
        inside = torch.ones(pts.shape[:2], dtype=torch.bool, device=pts.device)
        for j in range(nrm.shape[2]):
            inside &= (pts * nrm[:, i, j][:, None, :]).sum(dim=-1) \
                <= h[:, i, j][:, None]
        inside_any |= inside.any(dim=1)
    return inside_any


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

    # Exact ray-segment test on a directly constructed one-theta c-space.
    # Along beam 0 (+x), the first obstacle is [0.6 m, 1.2 m) and the second
    # starts at 1.8 m, so the three distances are [0.6, 0.6, 0.6] m.
    interval_cs = torch.zeros(N, G, G, 1, dtype=torch.bool)
    interval_cs[0, R + 3:R + 6, R, 0] = True
    interval_cs[0, R + 9:R + 11, R, 0] = True
    intervals = cost_ray_intervals_3d(
        interval_cs, robot_cell, num_beams=4, h_res_deg=90.0,
        max_range=max_range, g_min=g_min, acc=g_min,
    )
    assert tuple(intervals.shape) == (N, 3, 4, 1)
    assert torch.isclose(intervals[0, 0, 0, 0], torch.tensor(0.6))
    assert torch.isclose(intervals[0, 1, 0, 0], torch.tensor(0.6))
    assert torch.isclose(intervals[0, 2, 0, 0], torch.tensor(0.6))
    assert torch.equal(
        intervals[0, :, 2, 0], torch.tensor([max_range, 0.0, 0.0])
    )

    # The interval count controls the output channels.  With five channels the
    # second obstacle width and the trailing free range are retained as well.
    intervals_5 = cost_ray_intervals_3d(
        interval_cs, robot_cell, num_beams=4, h_res_deg=90.0,
        max_range=max_range, g_min=g_min, acc=g_min, num_intervals=5,
    )
    assert tuple(intervals_5.shape) == (N, 5, 4, 1)
    assert torch.allclose(
        intervals_5[0, :, 0, 0],
        torch.tensor([0.6, 0.6, 0.6, 0.4, 2.8]),
        atol=1e-5,
    )

    intervals_1 = cost_ray_intervals_3d(
        interval_cs, robot_cell, num_beams=4, h_res_deg=90.0,
        max_range=max_range, g_min=g_min, acc=g_min, num_intervals=1,
    )
    assert tuple(intervals_1.shape) == (N, 1, 4, 1)
    assert torch.isclose(intervals_1[0, 0, 0, 0], torch.tensor(0.6))

    # If no second obstacle is visible, channel 2 is the remaining free range.
    one_obstacle_cs = torch.zeros_like(interval_cs)
    one_obstacle_cs[0, R + 3:R + 6, R, 0] = True
    one_obstacle = cost_ray_intervals_3d(
        one_obstacle_cs, robot_cell, num_beams=4, h_res_deg=90.0,
        max_range=max_range, g_min=g_min, acc=g_min,
    )
    assert torch.allclose(
        one_obstacle[0, :, 0, 0], torch.tensor([0.6, 0.6, 3.8]),
        atol=1e-5,
    )

    proc = preprocess_cost_ray_intervals(
        intervals, max_range=max_range, epsilon=g_min / 2.0
    )
    assert torch.allclose(proc * max_range, intervals, atol=1e-5)

    radial = torch.arange(0.0, max_range, g_min)
    reconstructed = (
        (radial >= intervals[0, 0, 0, 0])
        & (radial < intervals[0, 0, 0, 0] + intervals[0, 1, 0, 0])
    )
    expected = (radial >= 0.6) & (radial < 1.2)
    assert torch.equal(reconstructed, expected)

    # ------------------------------------------------------------------
    # Analytic point-set backend (grid-free).  Sparrow geometry throughout:
    # 1.2 x 0.14 m footprint, 36 beams at 10 deg, 30 theta layers.
    # ------------------------------------------------------------------
    L, W = 1.2, 0.14
    nzs, as_ = 30, 2 * math.pi / 30
    beams = 36
    hres = 360.0 / beams
    polys = rectangles_to_polygons([(L, W, 0.0)])
    nrm, hoff, cfac = footprint_constraints(
        polys, num_beams=beams, nz=nzs, a=as_, h_res_degrees=hres
    )
    INF = float(torch.finfo(torch.float32).max)

    def ray_entry(points, nrm_, h_, c_, a_, beam, layer):
        """(M, 2) points -> entry distance for one (beam, layer)."""
        pts = points.view(1, -1, 2)
        return pointset_intervals(pts, nrm_, h_, c_, a_)[0][0, beam, layer].item()

    # A.1 axis-aligned single point.  Driving +x at heading 0, the robot meets
    # the point's own footprint-mirror [2.0 - 0.6, 2.0 + 0.6] -- the same
    # interval you get by hand from the four edge inequalities.
    assert abs(ray_entry(torch.tensor([[2.0, 0.0]]), nrm, hoff, cfac, as_,
                         0, 0) - 1.4) < 1e-5
    print("point-set A.1: entry 1.4000 (axis-aligned, 45 deg below)")

    # A.1b 45 degrees.  The tempting closed form `2 - h_F(u)` gives 1.5262 and
    # is WRONG -- the support point is not on the ray.  Intersecting the four
    # edge bounds gives [1.9010, 2.0990] instead.
    polys45 = rectangles_to_polygons([(L, W, 0.0)])
    n45, h45, c45 = footprint_constraints(
        polys45, num_beams=8, nz=nzs, a=as_, h_res_degrees=45.0
    )
    p45 = 2.0 * torch.tensor([[math.cos(math.pi / 4), math.sin(math.pi / 4)]])
    e45, x45 = pointset_intervals(p45.view(1, 1, 2), n45, h45, c45, as_)
    assert abs(e45[0, 1, 0].item() - 1.9010) < 1e-3, e45[0, 1, 0].item()
    assert abs(x45[0, 1, 0].item() - 2.0990) < 1e-3, x45[0, 1, 0].item()
    # ...and the beam along +x sees nothing: the geometry is direction-aware.
    assert e45[0, 0, 0].item() > 1e30
    print("point-set A.1b: 45 deg interval [1.9010, 2.0990], "
          "h_F(u) shortcut 1.5262 is wrong")

    # A.2 Lateral band.  A point 2.0 m ahead is only ever touched while its
    # lateral offset stays inside the footprint half-width W/2 = 0.07, and the
    # transition is *exactly* there (closed interval: grazing still touches).
    for d, touched in ((0.0, True), (0.03, True), (0.07, True),
                       (0.0701, False), (0.09, False), (0.30, False)):
        v = ray_entry(torch.tensor([[2.0, d]]), nrm, hoff, cfac, as_, 0, 0)
        if touched:
            assert abs(v - 1.4) < 1e-5, (d, v)
        else:
            assert v > 1e30, (d, v)
    # Two points straddling the axis: the corridor between their blobs opens at
    # s = 0.14 = W exactly, and its width is s - W (photon: 0.105 overlaps,
    # 0.28 leaves 0.14 m).  At 36 beams / 10 deg the neighbours at 1.6 m are
    # 0.279 m apart, which is ABOVE W -- so v1 still sees a 0.14 m phantom
    # corridor there.  That is a sampling-density limit, not a geometry bug.
    for s, touched in ((0.105, True), (0.14, True), (0.28, False)):
        v = ray_entry(torch.tensor([[2.0, s / 2], [2.0, -s / 2]]),
                      nrm, hoff, cfac, as_, 0, 0)
        if touched:
            assert abs(v - 1.4) < 1e-5, (s, v)
        else:
            assert v > 1e30, (s, v)
    print("point-set A.2: band edge at W/2 = 0.07, corridor opens at s = W")

    # A.3 Collision identity: the t = 0 case of the same inequalities.  Both
    # sides are tested per point, so neither "always true" nor "always false"
    # can pass.
    inside_pts = torch.tensor([[0.0, 0.0], [0.3, 0.0], [0.5, 0.05],
                               [0.0, 0.06], [-0.55, -0.06]])
    outside_pts = torch.tensor([[2.0, 0.0], [0.0, 0.2], [1.5, 1.5],
                                [-1.0, 0.0], [0.0, -0.5], [0.62, 0.0]])
    both_pts = torch.cat([inside_pts, outside_pts])
    expected = torch.cat([torch.ones(len(inside_pts), dtype=torch.bool),
                          torch.zeros(len(outside_pts), dtype=torch.bool)])
    per_point = torch.stack([
        pointset_collision(pt.view(1, 1, 2), nrm, hoff)[0] for pt in both_pts
    ])
    assert torch.equal(per_point, expected), (per_point, expected)

    # Whole clouds: one containing inside points collides, one without does not,
    # and the *entry* form of the same geometry agrees with both.
    for cloud, want in ((both_pts[None], True), (outside_pts[None], False)):
        hit = pointset_collision(cloud, nrm, hoff)
        from_entry = pointset_intervals(cloud, nrm, hoff, cfac, as_)[0][
            :, :, 0].min(dim=1).values == 0
        assert torch.equal(hit, from_entry), (hit, from_entry)
        assert hit.item() is want, (hit.item(), want)
    print("point-set A.3: pointset_collision == min_b entry(b, k=0) == 0, "
          f"per point ({len(inside_pts)} in / {len(outside_pts)} out)")

    # A.4 Endpoints: a point under the robot is an immediate contact on every
    # beam; a cloud entirely out of range is clear on every beam.
    under = pointset_intervals(torch.zeros(1, 1, 2), nrm, hoff, cfac, as_)[0]
    assert torch.equal(under[0, :, :], torch.zeros_like(under[0]))
    far = torch.full((1, 4, 2), 30.0)
    e_far, w_far, c_far = pointset_batch(far, nrm, hoff, cfac, as_, max_range)
    assert torch.all(e_far == max_range) and torch.all(w_far == 0.0)
    assert not c_far.any()
    print("point-set A.4: under-robot -> 0, out-of-range -> max_range, w=0")

    # A.5 Shape independence: a convex triangle, and a two-piece union where a
    # point sits in piece 2 only (pieces must OR, not intersect).
    tri = np.array([[0.6, 0.07], [-0.6, 0.07], [0.0, -0.07]])
    n_t, h_t, c_t = footprint_constraints([tri], num_beams=beams, nz=nzs,
                                         a=as_, h_res_degrees=hres)
    assert pointset_collision(torch.tensor([[[0.0, 0.0]]]), n_t, h_t).item()
    assert not pointset_collision(torch.tensor([[[0.0, 0.2]]]), n_t, h_t).item()
    assert pointset_collision(torch.tensor([[[0.3, 0.02]]]), n_t, h_t).item()
    e_tri = ray_entry(torch.tensor([[2.0, 0.0]]), n_t, h_t, c_t, as_, 0, 0)
    assert 1.0 < e_tri < 2.1, e_tri
    thin = rectangles_to_polygons([(L, W, 0.0)])
    both = rectangles_to_polygons([(L, W, 0.0), (0.3, 0.3, math.pi / 4)])
    n_b, h_b = polygon_constraints(both)
    n_b, h_b = torch.as_tensor(np.stack(n_b)), torch.as_tensor(np.stack(h_b))
    probe = torch.tensor([[[0.0, 0.15]]])      # in the 45 deg square only
    assert pointset_collision(probe, n_b, h_b).item()
    assert not pointset_collision(probe, nrm, hoff).item()
    print(f"point-set A.5: triangle entry {e_tri:.4f}; piece union ORs correctly")

    # A.6 Cross-backend consistency: the analytic test must reproduce the
    # rasterizer it replaces, cell centre by cell centre.  This is what kills a
    # whole class of frame/winding sign errors (the robot frame here is y-down
    # and the rasterizer rotates by -psi, so the sign cannot be reasoned out).
    g_p, half_p = 0.1, 35
    rectangles = [(L, W, 0.35)]                 # a rotated rectangle, on purpose
    masks = make_rectangle_union_masks(g_p, as_, nzs, rectangles, half_p,
                                       subsample=1)
    off = np.arange(-half_p, half_p + 1) * g_p
    xx, yy = np.meshgrid(off, off, indexing="ij")   # axis0 = x, axis1 = y
    centres = np.stack([xx.ravel(), yy.ravel()], axis=1)
    n_r, h_r, _ = footprint_constraints(rectangles_to_polygons(rectangles),
                                        num_beams=4, nz=nzs, a=as_)
    n_r_np, h_r_np = n_r.numpy()[0], h_r.numpy()[0]
    for layer in (0, 7):
        th = layer * as_
        ct, st = math.cos(th), math.sin(th)
        q = np.stack([centres[:, 0] * ct + centres[:, 1] * st,
                      -centres[:, 0] * st + centres[:, 1] * ct], axis=1)
        # One point per environment: pointset_collision ORs over a cloud, so the
        # per-cell answer needs a (cells, 1, 2) point set, not one big cloud.
        analytic = pointset_collision(
            torch.as_tensor(q, dtype=torch.float32).view(-1, 1, 2), n_r, h_r
        ).numpy().reshape(xx.shape)
        raster = masks[layer].astype(bool)
        near = np.abs(q @ n_r_np.T - h_r_np).min(axis=1) < 1e-6
        near = near.reshape(xx.shape)
        assert np.array_equal(analytic[~near], raster[~near]), \
            f"layer {layer}: {(analytic != raster)[~near].sum()} mismatches"
        assert raster.sum() > 0 and analytic.sum() > 0
        print(f"point-set A.6: layer {layer} matches rasterizer on "
              f"{(~near).sum()}/{near.size} cell centres "
              f"({near.sum()} edge-centres excluded)")

    # A.7 Batched per-env tables (what SparrowCspaceEnv builds): two envs with
    # different footprints must give exactly what the per-env path gives, and
    # padding a 1-piece env up to 3 pieces must change nothing.  This is the
    # "random footprint" case, where every env draws its own union.
    env_footprints = [
        [(L, W, 0.0)],                                     # thin rectangle
        [(L, W, 0.35), (0.3, 0.3, math.pi / 4)],           # rotated union
    ]
    polys_per_env = [rectangles_to_polygons(r) for r in env_footprints]
    n_b2, h_b2, c_b2 = footprint_constraint_batch(
        polys_per_env, num_beams=beams, nz=nzs, a=as_, h_res_degrees=hres
    )
    assert n_b2.shape[0] == 2 and c_b2.shape[:2] == (2, 2), (n_b2.shape, c_b2.shape)
    pts_batch = torch.tensor([[[2.0, 0.0], [1.0, 0.3], [0.0, 0.0]],
                              [[2.0, 0.0], [1.0, 0.3], [0.0, 0.0]]])
    e_batch, w_batch, c_batch = pointset_batch(
        pts_batch, n_b2, h_b2, c_b2, as_, max_range
    )
    for i, rects in enumerate(env_footprints):
        n_i, h_i, c_i = footprint_constraints(rectangles_to_polygons(rects),
                                             num_beams=beams, nz=nzs, a=as_,
                                             h_res_degrees=hres)
        e_i, w_i, c_i_hit = pointset_batch(pts_batch[i:i + 1], n_i, h_i, c_i,
                                           as_, max_range)
        assert torch.equal(e_batch[i], e_i[0]), i
        assert torch.equal(w_batch[i], w_i[0]), i
        assert c_batch[i].item() == c_i_hit[0].item(), i
    # Same clouds, same geometry, no padding (P = 1 vs the padded P = 2).
    n_pad, h_pad, c_pad = footprint_constraint_batch(
        [polys_per_env[0]], num_beams=beams, nz=nzs, a=as_, h_res_degrees=hres
    )
    n_flat, h_flat, c_flat = footprint_constraints(
        polys_per_env[0], num_beams=beams, nz=nzs, a=as_, h_res_degrees=hres
    )
    assert n_pad.shape[1] == n_flat.shape[0] == 1      # nothing to pad here
    assert torch.equal(n_pad[0], n_flat) and torch.equal(h_pad[0], h_flat)
    # Explicit padding: zero normal + h = -1 must read as "no contact" and
    # "no collision", never as a full-length free interval.
    pad_pts = torch.zeros(1, 2, 2)
    e_one = pointset_batch(pad_pts, n_flat.unsqueeze(0), h_flat.unsqueeze(0),
                           c_flat.unsqueeze(0), as_, max_range)[0]
    e_pad = pointset_batch(pad_pts, n_pad, h_pad, c_pad, as_, max_range)[0]
    assert torch.equal(e_one, e_pad)
    print("point-set A.7: batched/padded env tables match the per-env path")

    print("self-test OK")
