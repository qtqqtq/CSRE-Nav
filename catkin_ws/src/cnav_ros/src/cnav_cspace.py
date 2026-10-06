#!/usr/bin/env python

"""c-space observation helpers for cnav_ros -- vendored, no training tree needed.

These are copies of the pure-numpy/torch functions in
`DRL-robot-navigation/SAC/cspace_cost_rays.py` that the node actually calls, so
`cnav_ros` builds its observation without importing anything from a SAC
checkout.  The training env remains the single source of truth for what the
observation *means*; this file is the deployment copy of it.

Both geometry backends are kept, and the node picks one with `~backend`:

  - **grid** -- `make_footprint_masks` + `build_cspace` +
    `cost_ray_intervals_3d`: rasterize the footprint, dilate the occupancy grid
    to a 3D c-space, then march rays through it.  The original backend.
  - **points** -- `footprint_constraints` + `pointset_batch`: an obstacle is a
    lidar hit *point*, and "would driving along beam b at heading theta_k touch
    that point?" is answered in closed form, with no grid to quantise the
    surface onto.

The two agree on shapes and semantics and disagree on geometry, which is why
they are both here: a checkpoint must be deployed on the backend it was trained
with.  `SparrowCspaceEnv.__init__` carries the same branch.

Deliberately not copied (unused by the node, which has one fixed footprint and
never trains): `make_rectangle_union_masks`, `rectangles_to_polygons`,
`footprint_constraint_batch`, `cost_rays_3d`, `compute_cost_rays`,
`preprocess_cost_rays`, and the benchmark harness.  Behaviour of everything
below is unchanged from the training copies.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


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
