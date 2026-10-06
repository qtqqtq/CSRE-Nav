#!/usr/bin/env python

"""c-space observation helpers for cnav_ros -- vendored, no training tree needed.

A verbatim copy of `DRL-robot-navigation/SAC/cspace_cost_rays.py`, minus the
module docstring above and the `__main__` self-test: the node builds its
observation from this file, so `cnav_ros` needs no SAC checkout (`src/` and
`model/` make the package self-contained).  The training tree stays the single
source of truth for what the observation *means* -- re-sync this file by hand
after any change there.

An obstacle is a lidar hit *point*, and "would driving along beam b at heading
theta_k touch it?" is answered in closed form: no occupancy grid, no rasterized
c-space, no march.  The obstacle's own Minkowski footprint is intersected
against the robot's, one half-plane inequality per footprint edge, which keeps
obstacle positions at full metric precision.
"""
from __future__ import annotations

import math

import numpy as np
import torch

__all__ = [
    "preprocess_cost_ray_intervals",
    "rectangles_to_polygons",
    "polygon_constraints",
    "footprint_constraints",
    "footprint_constraint_batch",
    "pointset_intervals",
    "pointset_batch",
    "pointset_collision",
]


def preprocess_cost_ray_intervals(intervals, max_range=5.0):
    """Normalize ray-segment distance channels to ``[0, 1]``.

    Input ``(N, C, B, nz)`` is in metres, where ``C`` is the configured number
    of consecutive radial intervals.  Dividing every distance by ``max_range``
    gives the CNN a common numeric scale.
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
# Cost rays from a point set: obstacle = lidar hit point
# ---------------------------------------------------------------------------
# An obstacle is a point, not a cell.  A raster puts surfaces on `g_min` cell
# centres, and two adjacent hit cells can each round the *wrong* way, opening a
# phantom gap in a wall that should be closed; a point set has no such failure
# and keeps the hits at full metric precision.
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
def rectangles_to_polygons(rectangles):
    """``(length_m, width_m, angle_rad)`` rows -> convex (V, 2) polygons.

    The axis-aligned box ``|x| <= L/2, |y| <= W/2`` rotated by ``+angle`` in the
    robot frame, one convex piece per rectangle, in the same convention the
    env's random-footprint generator writes its ``(L, W, A)`` rows.
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
    ``h_res_degrees`` must be the beam spacing the env actually scans with
    (i.e. 360/num_beams), otherwise the cost rays would not describe the beams
    the lidar reports.
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
    out the far side -- the "first occupied width" that channel 1 of the live
    observation holds.  ``collide (N,)`` is the current-pose test at theta = 0.
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

