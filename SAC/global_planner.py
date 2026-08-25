"""
Global graph-based planner in the configuration space (x, y, theta).

Implements the "Planning on the Graph" pipeline (paper Algorithm 1 + 2) over the
*same* 3-D configuration space that `SparrowCspaceEnv` builds for its cost-ray
observation (cspace_cost_rays.make_footprint_masks / build_cspace):

    1. construct the configuration space first -- dilate the world occupancy
       grid by the robot footprint into a (x, y, theta) c-space;
    2. sample valid configurations (i, j, k) in Cfree as graph vertices;
    3. add connectivity edges -- *movement* (straight segment obstacle-free at a
       fixed theta) and *rotation* (in-place theta switch, c-space vertical line
       free) -- matching the paper's movement / rotation primitives;
    4. BFS for reachability, then Dijkstra for the cost-optimal path
       (single-robot variant: edge cost = 2-D world distance x lambda_s,
       lambda_s left as the paper's formation-preference extension);
    5. simplify to a compact waypoint list (poses carry (x, y, theta)).

Coordinate conventions (identical to Sparrow / cspace_cost_rays):
    * world is in cm, pygame y-down;
    * voxel (i, j, k): i = floor(x / res), j = floor(y / res),
      k = floor(theta / a) mod nz, res = g_min * 100 cm;
    * voxel centre -> world: x = (i + 0.5) * res, y = (j + 0.5) * res.
    * occ_grid is indexed [x, y] (axis0 = x = forward at theta=0, axis1 = y),
      exactly the axis convention of `build_cspace`.

Design note (deviation from the observation footprint): the planner defaults to
`COLLISION_FOOTPRINT` (28 x 28 cm square ~= the robot's collision circle,
collision_trsd = 14 cm) rather than settings.ROBOT_FOOTPRINT (1.2 m) -- the
1.2 m footprint would block > half the 800 cm map and snap start/goal ~0.6 m
into open space, making plans near-useless.  `footprint` remains configurable.
"""
from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np
import torch

from cspace_cost_rays import build_cspace, make_footprint_masks

__all__ = [
    "COLLISION_FOOTPRINT",
    "build_world_occ_grid",
    "GraphPlanner",
    "PathResult",
    "draw_planned_path",
]

# 28 x 28 cm square, ~circumscribing the collision circle (collision_trsd=14 cm).
COLLISION_FOOTPRINT = [
    [0.14, 0.14], [0.14, -0.14], [-0.14, -0.14], [-0.14, 0.14],
]


def build_world_occ_grid(env, g_min):
    """Downsample the env's full obstacle bitmap into a world occupancy grid.

    Reads `env.world_occ_full` (persisted by `Sparrow._static_obstacle_init`,
    1 cm bool, indexed [x, y]) and ORs in the MapObs obstacle cells.  Returns a
    (H, W) bool array at `g_min` resolution, indexed [x_cell, y_cell] -- the
    same axis convention `build_cspace` expects (axis0 = x = forward @ theta=0).
    """
    res = int(round(g_min * 100.0))
    ws = env.window_size
    full = getattr(env, "world_occ_full", None)
    if full is None:
        full = np.zeros((ws, ws), dtype=bool)
    H = W = ws // res
    occ = full[: H * res, : W * res].reshape(H, res, W, res).any(axis=(1, 3))
    if getattr(env, "MapObs", None):
        pts = env.vec_map_obs_P_shaped[env.ri]            # (P, 2, 1)
        xs = pts[:, 0, 0].cpu().numpy()
        ys = pts[:, 1, 0].cpu().numpy()
        xs = np.clip(xs // res, 0, H - 1)
        ys = np.clip(ys // res, 0, W - 1)
        occ[xs, ys] = True
    return occ


class PathResult:
    """Output of `GraphPlanner.plan`."""
    __slots__ = ("poses", "cells", "total_cost")

    def __init__(self, poses, cells, total_cost):
        self.poses = poses          # list[(x_cm, y_cm, theta)], incl. true start/goal
        self.cells = cells          # list[(i, j, k)] graph cells of the raw path
        self.total_cost = total_cost  # sum of edge weights (cm * lambda_s)


class GraphPlanner:
    """Grid-stride graph planner over a 3-D c-space (Algorithm 2).

    Args:
        g_min:  c-space spatial resolution (m).  Default 0.1 (10 cm).
        nz:     number of theta layers.
        a:      theta resolution (rad).
        footprint: robot footprint polygon (m, +x forward, +y left).  Default
            COLLISION_FOOTPRINT (28 x 28 cm) -- see module docstring.
        stride_ij: vertex sampling step in (i, j) (voxels).
        stride_k:  vertex sampling step in k (theta layers).
        connect_dist_cm: max world distance for a movement edge.
        min_wp_spacing_cm: simplify keeps waypoints at least this far apart.
        turn_cost_cm: weight of one rotation edge (in-place 40 deg turn).
        lambda_s: edge-cost scale (single robot: 1.0; formation extension).
    """

    def __init__(self, g_min=0.1, nz=10, a=0.698, footprint=None,
                 stride_ij=2, stride_k=1, connect_dist_cm=60,
                 min_wp_spacing_cm=40, turn_cost_cm=5.0, lambda_s=1.0):
        self.g_min = g_min
        self.nz = nz
        self.a = a
        self.footprint = footprint if footprint is not None else COLLISION_FOOTPRINT
        self.stride_ij = stride_ij
        self.stride_k = stride_k
        self.connect_dist_cm = connect_dist_cm
        self.min_wp_spacing_cm = min_wp_spacing_cm
        self.turn_cost_cm = turn_cost_cm
        self.lambda_s = lambda_s
        self.res = int(round(g_min * 100.0))             # cm per voxel
        self.masks = make_footprint_masks(g_min, a, nz, self.footprint)

        # populated by build_graph
        self.cspace = None      # (H, W, nz) bool, 1 = in collision
        self.cfree = None
        self.verts = []         # list[(i, j, k)] -- persistent ids 0..nV_persistent-1
        self.adj = []           # adj[v] = list[(nbr, weight)]
        self.nV_persistent = 0

    # ------------------------------------------------------------------ cspace
    def build_graph(self, occ_grid):
        """Build the c-space, sample vertices and add movement / rotation edges."""
        occ_t = torch.as_tensor(np.asarray(occ_grid, dtype=bool),
                                dtype=torch.float32).unsqueeze(0)   # (1, H, W)
        cspace = build_cspace(occ_t, self.masks).squeeze(0).numpy()  # (H, W, nz) bool
        self.cspace = cspace
        self.cfree = ~cspace
        cfree = self.cfree
        H, W, nz = cspace.shape
        res, s_ij, s_k = self.res, self.stride_ij, self.stride_k

        si = list(range(0, H, s_ij))
        sj = list(range(0, W, s_ij))
        sk = list(range(0, nz, s_k))
        self.si, self.sj, self.sk = si, sj, sk

        # --- vertices: valid configurations (Cfree) on a regular grid stride ---
        verts = []
        vmap = np.full((len(si), len(sj), len(sk)), -1, dtype=np.int64)
        for ii, i in enumerate(si):
            for jj, j in enumerate(sj):
                for kk, k in enumerate(sk):
                    if cfree[i, j, k]:
                        vmap[ii, jj, kk] = len(verts)
                        verts.append((i, j, k))
        self.vmap = vmap
        adj = [[] for _ in verts]

        nI, nJ, nK = vmap.shape
        win = int(math.ceil(self.connect_dist_cm / (res * s_ij)))
        # distance filter over the (2*win+1)^2 window, in world cm
        di = np.arange(-win, win + 1, dtype=np.float64)[:, None]
        dj = np.arange(-win, win + 1, dtype=np.float64)[None, :]
        dmask = ((di * di + dj * dj) * (res * s_ij) ** 2
                 <= self.connect_dist_cm ** 2)
        dmask[win, win] = False                            # no self edges

        # --- movement edges (same theta, 3-D segment obstacle-free) ---
        pairs = []          # (i1, j1, i2, j2, k) int64
        pair_verts = []     # (v1, v2) parallel to pairs
        for kk in range(nK):
            k = sk[kk]
            for ii in range(nI):
                for jj in range(nJ):
                    v = int(vmap[ii, jj, kk])
                    if v < 0:
                        continue
                    i, j, _ = verts[v]
                    for ddi in range(-win, win + 1):
                        for ddj in range(-win, win + 1):
                            if not dmask[ddi + win, ddj + win]:
                                continue
                            ii2, jj2 = ii + ddi, jj + ddj
                            if not (0 <= ii2 < nI and 0 <= jj2 < nJ):
                                continue
                            v2 = int(vmap[ii2, jj2, kk])
                            if v2 < 0 or v2 <= v:
                                continue
                            i2, j2, _ = verts[v2]
                            pairs.append((i, j, i2, j2, k))
                            pair_verts.append((v, v2))
        if pairs:
            free = self._segments_free(np.asarray(pairs, dtype=np.int64))
            for idx, ok in enumerate(free):
                if ok:
                    v1, v2 = pair_verts[idx]
                    i1, j1, i2, j2, _ = pairs[idx]
                    d = math.hypot((i2 - i1) * res, (j2 - j1) * res) * self.lambda_s
                    adj[v1].append((v2, d))
                    adj[v2].append((v1, d))

        # --- rotation edges (in-place adjacent-theta switch, vertical line free) ---
        for ii in range(nI):
            i = si[ii]
            for jj in range(nJ):
                j = sj[jj]
                for kk in range(nK):
                    v = int(vmap[ii, jj, kk])
                    if v < 0:
                        continue
                    kk2 = (kk + 1) % nK
                    v2 = int(vmap[ii, jj, kk2])
                    if v2 >= 0 and v2 != v:
                        w = self.turn_cost_cm * self.lambda_s
                        adj[v].append((v2, w))
                        adj[v2].append((v, w))

        self.verts = verts
        self.adj = adj
        self.nV_persistent = len(verts)
        return len(verts)

    # -------------------------------------------------------------- segment DDA
    def _segments_free(self, pairs):
        """Vectorized segment-free check in the 3-D c-space.

        `pairs`: (P, 5) int64 array of (i1, j1, i2, j2, k).  Returns (P,) bool;
        a segment is free iff every voxel it passes through (round-based unit
        sampling, the same DDA pattern as `cost_rays_3d`) is in Cfree and stays
        inside the map.
        """
        cs = self.cspace                                     # (H, W, nz) bool
        H, W, nz = cs.shape
        i1, j1, i2, j2, kk = pairs.T
        di, dj = i2 - i1, j2 - j1
        L = max(1, int(np.abs(di).max()), int(np.abs(dj).max()))
        t = (np.arange(1, L + 1)) / L                        # (L,)
        ci = np.round(i1[:, None] + di[:, None] * t[None, :]).astype(np.int64)
        cj = np.round(j1[:, None] + dj[:, None] * t[None, :]).astype(np.int64)
        inb = (ci >= 0) & (ci < H) & (cj >= 0) & (cj < W)
        occ = cs[np.clip(ci, 0, H - 1), np.clip(cj, 0, W - 1), kk[:, None]]
        return ~occ.any(axis=1) & inb.all(axis=1)

    # -------------------------------------------------------------- coordinates
    def world_to_voxel(self, x_cm, y_cm, theta=None):
        """World (cm) -> voxel (i, j, k).  theta=None -> k=None (any layer)."""
        res, H, W = self.res, self.cspace.shape[0], self.cspace.shape[1]
        i = min(max(int(x_cm // res), 0), H - 1)
        j = min(max(int(y_cm // res), 0), W - 1)
        if theta is None:
            return i, j, None
        k = int((theta % (2 * math.pi)) / self.a) % self.nz
        return i, j, k

    def voxel_to_world(self, cell):
        i, j, k = cell
        return (i + 0.5) * self.res, (j + 0.5) * self.res, k * self.a

    def _snap_to_free(self, pose):
        """Nearest Cfree voxel to `pose` (radius-first over (i,j), all theta).

        Returns a (i, j, k) cell guaranteed free, or None if the whole map is
        unreachable from the pose's neighbourhood (fully enclosed).
        """
        x, y, theta = pose
        i, j, k0 = self.world_to_voxel(x, y, theta)
        cfree = self.cfree
        H, W, nz = cfree.shape
        if k0 is not None and cfree[i, j, k0]:
            return i, j, k0
        rings = range(1, max(H, W))
        for r in rings:
            for di in range(-r, r + 1):
                ii = i + di
                if not (0 <= ii < H):
                    continue
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    jj = j + dj
                    if not (0 <= jj < W):
                        continue
                    if k0 is None:
                        for k in range(nz):
                            if cfree[ii, jj, k]:
                                return ii, jj, k
                    else:
                        for kk in range(nz):           # prefer the layer nearest k0
                            k = (k0 + kk) % nz
                            if cfree[ii, jj, k]:
                                return ii, jj, k
        return None

    def _ensure_vertex(self, cell, added):
        """Add `cell` as a temporary vertex, connecting it to every existing
        vertex within connect_dist (movement, same theta) / adjacent theta
        (rotation).  Returns its id; records newly-added ids in `added`."""
        for v, c in enumerate(self.verts):
            if c[0] == cell[0] and c[1] == cell[1] and c[2] == cell[2]:
                return v
        i, j, k = cell
        vid = len(self.verts)
        self.verts.append(cell)
        self.adj.append([])
        added.append(vid)

        move_nbrs = []          # (v2, i2, j2, d)
        rot_nbrs = []           # (v2, w)
        for v2, (i2, j2, k2) in enumerate(self.verts):
            if v2 == vid:
                continue
            if k2 == k:
                d = math.hypot((i2 - i) * self.res, (j2 - j) * self.res)
                if 0 < d <= self.connect_dist_cm:
                    move_nbrs.append((v2, i2, j2, d))
            elif (i2, j2) == (i, j) and ((k2 - k) % self.nz) in (1, self.nz - 1):
                rot_nbrs.append((v2, self.turn_cost_cm))

        if move_nbrs:
            pairs = np.asarray([[i, j, i2, j2, k] for (_, i2, j2, _) in move_nbrs],
                               dtype=np.int64)
            free = self._segments_free(pairs)
            for (v2, _, _, d), ok in zip(move_nbrs, free):
                if ok:
                    self.adj[vid].append((v2, d * self.lambda_s))
                    self.adj[v2].append((vid, d * self.lambda_s))
        for v2, w in rot_nbrs:
            self.adj[vid].append((v2, w * self.lambda_s))
            self.adj[v2].append((vid, w * self.lambda_s))
        return vid

    def _remove_extra(self, count):
        """Drop the last `count` (temporary) vertices and all their edges."""
        if count <= 0:
            return
        verts, adj = self.verts, self.adj
        n = len(verts)
        lo = n - count
        for vid in range(lo, n):
            for nbr, _ in adj[vid]:
                lst = adj[nbr]
                for idx, (v, _w) in enumerate(lst):
                    if v == vid:
                        lst.pop(idx)
                        break
        del verts[lo:n]
        del adj[lo:n]

    # ------------------------------------------------------------------ search
    def _bfs_connected(self, s, g):
        """Unweighted BFS connectivity between vertex ids s and g."""
        seen = {s}
        q = deque([s])
        while q:
            u = q.popleft()
            if u == g:
                return True
            for v, _ in self.adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return False

    def _dijkstra(self, s, g):
        """Dijkstra with early stop at g.  Returns (path_ids, cost) or None."""
        n = len(self.verts)
        INF = float("inf")
        dist = [INF] * n
        prev = [-1] * n
        dist[s] = 0.0
        pq = [(0.0, s)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == g:
                break
            for v, w in self.adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if dist[g] == INF:
            return None
        path = []
        u = g
        while u != -1:
            path.append(u)
            u = prev[u]
        path.reverse()
        return path, dist[g]

    # ------------------------------------------------------------------- public
    def is_connected(self, start_pose, goal_pose):
        """Reachability only (BFS), no path reconstruction."""
        if self.cspace is None:
            return False
        s = self._snap_to_free(start_pose)
        g = self._snap_to_free(goal_pose)
        if s is None or g is None:
            return False
        added = []
        s_id = self._ensure_vertex(s, added)
        g_id = self._ensure_vertex(g, added)
        try:
            return self._bfs_connected(s_id, g_id)
        finally:
            self._remove_extra(len(added))

    def plan(self, start_pose, goal_pose):
        """Plan a path from `start_pose` to `goal_pose` (x_cm, y_cm, theta).

        Returns a `PathResult` or None when no feasible path exists (or the
        graph has not been built yet).  `goal_pose`'s theta may be None (any
        orientation at the goal).
        """
        if self.cspace is None:
            return None
        s = self._snap_to_free(start_pose)
        g = self._snap_to_free(goal_pose)
        if s is None or g is None:
            return None
        added = []
        s_id = self._ensure_vertex(s, added)
        g_id = self._ensure_vertex(g, added)
        try:
            if not self._bfs_connected(s_id, g_id):
                return None
            found = self._dijkstra(s_id, g_id)
            if found is None:
                return None
            path, cost = found
            return self._to_result(path, start_pose, goal_pose, cost)
        finally:
            self._remove_extra(len(added))

    def _to_result(self, path, start_pose, goal_pose, cost):
        """Graph vertex ids -> (start, ...waypoints..., goal) world poses."""
        verts = self.verts
        cells = [verts[v] for v in path]
        poses = [start_pose]
        for v in path[1:-1]:
            poses.append(self.voxel_to_world(verts[v]))
        poses.append(goal_pose)
        return PathResult(self._simplify(poses), cells, cost)

    def _simplify(self, poses):
        """Drop intermediate waypoints closer than min_wp_spacing_cm to the
        previous one (keeps the endpoints; the SAC policy does local avoidance
        between waypoints)."""
        if len(poses) <= 2:
            return list(poses)
        out = [poses[0]]
        for p in poses[1:-1]:
            if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) >= self.min_wp_spacing_cm:
                out.append(p)
        out.append(poses[-1])
        return out


def draw_planned_path(env, poses):
    """Draw the planned waypoint polyline (magenta) onto `env.canvas`.

    Waypoint dots + per-waypoint heading ticks.  No-op when not rendered.
    """
    import pygame
    canvas = getattr(env, "canvas", None)
    if canvas is None or not poses:
        return

    def _gd(x, y):
        g = env._world_2_grid(torch.tensor([float(x), float(y)]))
        return (int(g[0]), int(g[1]))

    pts = [_gd(x, y) for (x, y, _) in poses]
    for a, b in zip(pts[:-1], pts[1:]):
        pygame.draw.line(canvas, (255, 0, 255), a, b, width=4)
    for p in pts[1:-1]:
        pygame.draw.circle(canvas, (255, 0, 255), p, 4)
    for (x, y, t) in poses:
        a = _gd(x, y)
        b = _gd(x + 12 * math.cos(t), y - 12 * math.sin(t))
        pygame.draw.line(canvas, (255, 170, 255), a, b, width=1)


# ---------------------------------------------------------------------------
# Self-test (run inside the container: `python3 global_planner.py`)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def test_open_map():
        H = W = 80
        occ = np.zeros((H, W), dtype=bool)
        pl = GraphPlanner()
        pl.build_graph(occ)
        res = pl.plan((100, 100, 0.0), (700, 700, 1.0))
        assert res is not None, "open map should be connected"
        assert res.poses[0] == (100, 100, 0.0)
        assert res.poses[-1] == (700, 700, 1.0)
        d = math.hypot(600, 600)
        assert d <= res.total_cost <= d * 1.6, "%.1f vs euclid %.1f" % (
            res.total_cost, d)
        # every graph-cell waypoint must be in Cfree at its own theta
        for (i, j, k) in res.cells:
            assert not pl.cspace[i, j, k]
        print("open-map OK: cost %.1f (euclid %.1f), %d waypoints"
              % (res.total_cost, d, len(res.poses)))

    def test_obstacle_island():
        H = W = 80
        occ = np.zeros((H, W), dtype=bool)
        occ[40, 5:75] = True                      # vertical wall, blocks x=40
        pl = GraphPlanner()
        pl.build_graph(occ)
        start, goal = (100, 200, 0.0), (700, 600, 0.0)
        res = pl.plan(start, goal)
        assert res is not None, "wall has open ends -> detour exists"
        d = math.hypot(600, 400)
        assert res.total_cost > d * 1.05, "must detour around the wall"
        for (i, j, k) in res.cells:
            assert not pl.cspace[i, j, k]
        print("obstacle-island OK: cost %.1f (straight %.1f), %d waypoints"
              % (res.total_cost, d, len(res.poses)))

    def test_enclosed_goal():
        H = W = 80
        occ = np.zeros((H, W), dtype=bool)
        occ[58:72, 55:72] = True      # hollow box walls (2 voxels thick)
        occ[60:70, 57:70] = False     # interior, enclosing voxel (62, 62)
        pl = GraphPlanner()
        pl.build_graph(occ)
        res = pl.plan((100, 100, 0.0), (620, 620, 0.0))
        assert res is None, "goal inside a fully-enclosed box -> no path"
        # sanity: start is reachable in open space, goal voxel is free (inside box)
        assert pl.plan((100, 100, 0.0), (100, 400, 0.0)) is not None
        assert not pl.cspace[62, 62, 0], "goal voxel free inside the box"
        print("enclosed-goal OK: plan() returns None")

    def test_theta_gap():
        # 40 x 14 cm footprint: a 3-voxel gap (30 cm) in a 1-voxel wall is
        # passable only when the robot's LONG axis is NOT along x (theta ~ 90).
        H = W = 40
        occ = np.zeros((H, W), dtype=bool)
        occ[:, 20] = True                         # horizontal wall row y=20
        occ[18:21, 20] = False                    # gap x in [18,21) -> 30 cm
        fp = [[0.20, 0.07], [0.20, -0.07], [-0.20, -0.07], [-0.20, 0.07]]
        pl = GraphPlanner(footprint=fp, stride_ij=1)
        pl.build_graph(occ)
        # c-space theta-dependence: gap cell (19,20) blocked at theta=0, free at k=2 (~80 deg)
        assert pl.cspace[19, 20, 0], "long axis along x must NOT fit the 30 cm gap"
        assert not pl.cspace[19, 20, 2], "rotated ~80 deg must fit the gap"
        res = pl.plan((50, 100, 0.0), (350, 300, 0.0))
        assert res is not None, "must thread the gap via rotation"
        crossed_rotated = any(abs(math.cos(t)) < 0.9 for (_, _, t) in res.poses[1:-1])
        assert crossed_rotated, "path must rotate near the gap, got theta: %s" % [
            "%.0f" % math.degrees(t) for (_, _, t) in res.poses]
        for (i, j, k) in res.cells:
            assert not pl.cspace[i, j, k]
        print("theta-gap OK: c-space theta-dimension enforced, %d waypoints"
              % len(res.poses))

    test_open_map()
    test_obstacle_island()
    test_enclosed_goal()
    test_theta_gap()
    print("all global_planner self-tests passed")
