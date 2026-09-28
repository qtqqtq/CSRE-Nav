#!/usr/bin/env python3

"""ROS-free geometry and world helpers for the benchmark scripts.

The current world has 20 moving obstacles. Fixed geometry is parsed from the
SDF file; moving model positions are read from Gazebo during each trial.
The standalone self-test checks that the copied world contains the expected
population and stable physics settings.
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import xml.etree.ElementTree as ET


def sha1_of(path):
    """Content hash of a file, used to pin the world a run was scored against.

    Lives here rather than in each script so the goal set, the harness and the
    report all mean the same thing by "the world hash".
    """
    with open(path, 'rb') as fh:
        return hashlib.sha1(fh.read()).hexdigest()


# --------------------------------------------------------------------------
# Vendored from SAC/laser_env.py -- keep byte-compatible where possible.
# --------------------------------------------------------------------------


def _parse_sdf_pose(pose_str):
    """Parse SDF pose string 'x y z roll pitch yaw' -> (x, y, z, roll, pitch, yaw)."""
    parts = [float(p) for p in pose_str.split()]
    while len(parts) < 6:
        parts.append(0.0)
    return tuple(parts[:6])


def _rotate_2d(x, y, yaw):
    """Rotate point (x, y) by yaw radians in the XY plane."""
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (x * c - y * s, x * s + y * c)


def parse_world_obstacles(world_path, margin=0.3):
    """Parse an SDF world file and extract axis-aligned bounding boxes of static obstacles.

    Walks every <model> -> <link> -> <collision> chain, resolves the full 6-DOF
    transform into the world frame, and returns a list of *(x_min, x_max, y_min,
    y_max)* tuples.

    Vendored from ``SAC/laser_env.py``; see the module docstring for why.

    Three traps worth restating, because the bench world is authored against them:

    1. ``root.iter('model')`` is **recursive** -- never nest a ``<model>`` inside
       another, or every inner collision is counted twice.
    2. ``collision.find('geometry')`` is a **direct child** lookup, so
       ``<geometry>`` must sit immediately under ``<collision>``.
    3. The skip list is a **substring** match on the model *name*.  Anything
       containing ``plane``/``ground``/``floor`` is dropped, so obstacle names
       must avoid those words and the floor must contain one of them.

    Args:
        world_path: path to the ``.world`` SDF file.
        margin:     extra padding (metres) added to every AABB.

    Returns:
        list[tuple[float, float, float, float]]  -- one AABB per collision box.
    """
    tree = ET.parse(world_path)
    root = tree.getroot()
    obstacles = []

    # Only world-level static models belong to the fixed collision map. The
    # migrated Limo world has a population and 10 moving models; their current
    # positions are read from /gazebo/model_states by bench.py instead.
    for model in root.findall('./world/model'):
        name = model.get('name', '')
        if model.findtext('static', '').strip().lower() in ('0', 'false'):
            continue
        if 'cardboard_box' in name:
            continue
        # Skip ground/floor plates (e.g. ground_plane, asphalt_plane): they are
        # drivable surfaces, not obstacles.  Treating their collision boxes as
        # obstacles would mark the whole navigable floor as "inside an obstacle".
        if 'plane' in name or 'ground' in name or 'floor' in name:
            continue

        # Model-level pose (world frame)
        model_pose_elem = model.find('pose')
        model_pose = _parse_sdf_pose(
            model_pose_elem.text if model_pose_elem is not None else '0 0 0 0 0 0')
        mx, my, _, _, _, myaw = model_pose

        for link in model.iter('link'):
            link_pose_elem = link.find('pose')
            link_pose = _parse_sdf_pose(
                link_pose_elem.text if link_pose_elem is not None else '0 0 0 0 0 0')
            lx, ly, _, _, _, lyaw = link_pose

            for collision in link.iter('collision'):
                coll_pose_elem = collision.find('pose')
                coll_pose = _parse_sdf_pose(
                    coll_pose_elem.text if coll_pose_elem is not None else '0 0 0 0 0 0')
                cx, cy, _, _, _, cyaw = coll_pose

                geometry = collision.find('geometry')
                if geometry is None:
                    continue

                # ---- Determine half-extents in collision-local XY frame ----
                half_x, half_y = None, None  # will be set by the geometry handler

                box = geometry.find('box')
                cylinder = geometry.find('cylinder')
                mesh = geometry.find('mesh')

                if box is not None:
                    size_elem = box.find('size')
                    if size_elem is None:
                        continue
                    sx, sy, _sz = [float(v) for v in size_elem.text.split()]
                    half_x, half_y = sx / 2.0, sy / 2.0
                elif cylinder is not None:
                    radius_elem = cylinder.find('radius')
                    if radius_elem is None:
                        continue
                    r = float(radius_elem.text)
                    # A circle is over-approximated by its bounding square: the
                    # corners gain up to (sqrt(2)-1)*r ~= 0.41r.  Kept because it
                    # is the convention the training env used, so goal rejection
                    # here matches goal rejection there.
                    half_x, half_y = r, r
                elif mesh is not None:
                    # Mesh geometry cannot be reliably reduced to an AABB;
                    # skip it (rare -- typically only small decorative models).
                    continue
                else:
                    continue  # sphere, plane, etc. -- ignore

                if half_x is None:
                    continue

                # Compose transforms: model -> link -> collision -> box corners
                total_yaw = myaw + lyaw + cyaw

                # World position of collision origin
                #   p_world = R(myaw) * (R(lyaw) * [cx,cy] + [lx,ly]) + [mx,my]
                cx_rot, cy_rot = _rotate_2d(cx, cy, lyaw)
                wx = mx + _rotate_2d(lx + cx_rot, ly + cy_rot, myaw)[0]
                wy = my + _rotate_2d(lx + cx_rot, ly + cy_rot, myaw)[1]

                # Corners in collision-local frame, rotated to world
                corners = [
                    (-half_x, -half_y), (half_x, -half_y),
                    (-half_x, half_y),  (half_x, half_y),
                ]
                world_corners = [_rotate_2d(cx2, cy2, total_yaw) for cx2, cy2 in corners]
                xs = [wx + wcx for wcx, _ in world_corners]
                ys = [wy + wcy for _, wcy in world_corners]

                obstacles.append((
                    min(xs) - margin, max(xs) + margin,
                    min(ys) - margin, max(ys) + margin,
                ))

    return obstacles


def check_pos(x, y, obstacles=None, boundary=4.5, bounds=None, margin=0.0):
    """Return True if *(x, y)* is free of obstacles and within the world boundary.

    Vendored from ``SAC/laser_env.py`` (the legacy hard-coded TD3.world branch is
    kept for signature compatibility but is never used by the bench, which always
    passes an explicit ``bounds``).

    Args:
        x, y:       position to check.
        obstacles:  list of ``(x_min, x_max, y_min, y_max)`` AABBs, or **None**
                    to use the legacy hard-coded rectangles for TD3.world.
        boundary:   absolute x/y limit.  Points outside ``[-boundary, boundary]``
                    are rejected (used only when ``bounds`` is None).
        bounds:     optional ``(x_min, x_max, y_min, y_max)`` rectangle; when
                    given, it replaces the symmetric boundary check (for worlds
                    whose map is not centred at the origin).
        margin:     extra clearance (m) kept between the point and each obstacle
                    AABB.  Used for goal placement so goals never land right on
                    an obstacle edge.
    """
    # Boundary check
    if bounds is not None:
        x_min, x_max, y_min, y_max = bounds
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            return False
    elif abs(x) > boundary or abs(y) > boundary:
        return False

    if obstacles is None:
        # Legacy hard-coded obstacles for TD3.world (backward compatibility)
        if -3.8 > x > -6.2 and 6.2 > y > 3.8:
            return False
        if -1.3 > x > -2.7 and 4.7 > y > -0.2:
            return False
        if -0.3 > x > -4.2 and 2.7 > y > 1.3:
            return False
        if -0.8 > x > -4.2 and -2.3 > y > -4.2:
            return False
        if -1.3 > x > -3.7 and -0.8 > y > -2.7:
            return False
        if 4.2 > x > 0.8 and -1.8 > y > -3.2:
            return False
        if 4 > x > 2.5 and 0.7 > y > -3.2:
            return False
        if 6.2 > x > 3.8 and -3.3 > y > -4.2:
            return False
        if 4.2 > x > 1.3 and 3.7 > y > 1.5:
            return False
        if -3.0 > x > -7.2 and 0.5 > y > -1.5:
            return False
        return True

    # Dynamic check against parsed obstacle AABBs
    for (x_min, x_max, y_min, y_max) in obstacles:
        if (x_min - margin) < x < (x_max + margin) and (y_min - margin) < y < (y_max + margin):
            return False
    return True


# --------------------------------------------------------------------------
# New in planner_bench.
# --------------------------------------------------------------------------


def infer_bounds(obstacles, max_thickness=0.5):
    """Infer the interior (drivable) rectangle of a walled arena.

    Returns ``(x_min, x_max, y_min, y_max)`` -- the *inner* faces of the boundary
    walls, i.e. the extent a goal may legally occupy.

    Unlike the upstream heuristic this keys off wall *thinness* rather than
    ``span_y < 1.0`` after a 0.3 m pad, so it works for any wall up to
    ``max_thickness`` and does not depend on the caller passing a margin.
    Boundary walls are the AABBs that reach the global extent *and* are thin on
    the axis perpendicular to that side; a long wall lying along the same edge
    is therefore not mistaken for the edge itself.
    """
    if not obstacles:
        raise ValueError("no obstacles parsed -- wrong world file?")

    x_lo_ext = min(o[0] for o in obstacles)
    x_hi_ext = max(o[1] for o in obstacles)
    y_lo_ext = min(o[2] for o in obstacles)
    y_hi_ext = max(o[3] for o in obstacles)

    tol = 1e-6
    west = [o[1] for o in obstacles
            if abs(o[0] - x_lo_ext) < tol and (o[1] - o[0]) <= max_thickness]
    east = [o[0] for o in obstacles
            if abs(o[1] - x_hi_ext) < tol and (o[1] - o[0]) <= max_thickness]
    south = [o[3] for o in obstacles
             if abs(o[2] - y_lo_ext) < tol and (o[3] - o[2]) <= max_thickness]
    north = [o[2] for o in obstacles
             if abs(o[3] - y_hi_ext) < tol and (o[3] - o[2]) <= max_thickness]

    for name, side in (("west", west), ("east", east),
                       ("south", south), ("north", north)):
        if not side:
            raise ValueError(
                "cannot infer the %s boundary: no AABB reaches the arena extent "
                "with thickness <= %.2f m" % (name, max_thickness))

    return (max(west), min(east), max(south), min(north))


def point_aabb_distance(px, py, aabb):
    """Euclidean distance from a point to an AABB (0.0 when inside)."""
    dx = max(aabb[0] - px, 0.0, px - aabb[1])
    dy = max(aabb[2] - py, 0.0, py - aabb[3])
    return math.hypot(dx, dy)


def aabb_gap(a, b):
    """Euclidean distance between two AABBs (0.0 when they touch or overlap)."""
    dx = max(a[0] - b[1], b[0] - a[1], 0.0)
    dy = max(a[2] - b[3], b[2] - a[3], 0.0)
    return math.hypot(dx, dy)


# ------------------------------------------------------------------ footprint
# The robot's shape is a polygon of connected points, not a length x width box.
# That is how SAC training carries it: ``SAC/settings.py`` builds
# ``ROBOT_FOOTPRINT = rectangular_footprint(1.20, 0.14)`` as a list of ``[x, y]``
# vertices, and ``SAC/cspace_cost_rays.py`` consumes it as a generic ``(V, 2)``
# array, so neither side assumes four vertices or axis alignment.  Keeping the
# same representation here means a non-rectangular shape can be scored without
# inventing a second one, and it is what the laser collision test is expressed
# against: the polygon *is* the robot for scoring purposes.
#
# Points are in the robot frame (``base_footprint``), +x forward and +y left,
# centred on the origin -- the same convention as SAC's.
def parse_footprint(spec, what="footprint"):
    """Validate a footprint vertex list read from YAML -> [(x, y), ...]."""
    if not isinstance(spec, (list, tuple)) or len(spec) < 3:
        raise ValueError("%s needs at least 3 [x, y] points, got %r"
                         % (what, spec))
    pts = []
    for i, v in enumerate(spec):
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError("%s[%d] is %r; expected [x, y]" % (what, i, v))
        pts.append((float(v[0]), float(v[1])))
    if len(set(pts)) < 3:
        raise ValueError("%s has fewer than 3 distinct points: %r" % (what, pts))
    return pts


def polygon_extent(poly):
    """(length, width) of the polygon's axis-aligned bounding box, in metres."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (max(xs) - min(xs), max(ys) - min(ys))


def point_in_polygon(px, py, poly):
    """Even-odd ray cast: is (px, py) inside the closed polygon?

    The boundary counts as outside only in the sense that a point exactly on an
    edge may fall either way; nothing here depends on that case, because the
    silhouette below is continuous through it.
    """
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > py) != (y1 > py):
            if px < x0 + (py - y0) * (x1 - x0) / (y1 - y0):
                inside = not inside
    return inside


def polygon_ray_exit(px, py, theta, poly):
    """Distance from an interior point to the polygon boundary along ``theta``.

    This is the robot's own silhouette radius in that direction: how far a beam
    has to travel from ``(px, py)`` to leave its own footprint.  A laser return
    shorter than this terminated inside the robot, so what the beam hit was the
    footprint rather than an obstacle.

    Returns ``None`` when the ray never meets the boundary, which means the
    origin is outside the polygon; callers treat that as "no self-occlusion",
    since a laser that is not inside the footprint cannot be blocked by it.
    """
    dx, dy = math.cos(theta), math.sin(theta)
    best = None
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        ex, ey = x1 - x0, y1 - y0
        den = dx * ey - dy * ex
        if abs(den) < 1e-12:                      # ray parallel to this edge
            continue
        # p + t*d == v0 + u*e, solved by Cramer's rule for t along the ray and
        # u along the edge; the crossing is real only for t > 0 and u in [0, 1].
        wx, wy = x0 - px, y0 - py
        t = (wx * ey - wy * ex) / den
        u = (wx * dy - wy * dx) / den
        if t > 0.0 and -1e-9 <= u <= 1.0 + 1e-9:
            if best is None or t < best:
                best = t
    return best


def point_polygon_distance(px, py, poly):
    """Distance from a point to a closed polygon: 0.0 when inside, else the
    distance to the nearest edge."""
    if point_in_polygon(px, py, poly):
        return 0.0
    n = len(poly)
    best = float("inf")
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        ex, ey = x1 - x0, y1 - y0
        den = ex * ex + ey * ey
        if den <= 0.0:
            t = 0.0
        else:
            t = ((px - x0) * ex + (py - y0) * ey) / den
            t = min(1.0, max(0.0, t))
        best = min(best, math.hypot(px - (x0 + t * ex), py - (y0 + t * ey)))
    return best


def footprint_clearance(cx, cy, yaw, poly, models):
    """Signed clearance from the footprint, placed at ``(cx, cy, yaw)`` in the
    world, to the moving obstacles' bounding circles.

    ``models`` is an iterable of ``(name, x, y)`` ground-truth Gazebo poses.
    The obstacles are points in the world and the footprint is a polygon in the
    robot frame, so each obstacle is rotated into that frame and measured
    against the polygon there -- no second copy of the shape is needed.

    This is for questions asked *before* a teleport, like "is the start pose
    free right now": the laser cannot answer those, because it is still mounted
    on a robot that is somewhere else.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    best = float("inf")
    for name, x, y in models:
        radius = moving_obstacle_radius(name)
        if radius is None:
            continue
        dx, dy = x - cx, y - cy
        # World -> robot frame is a rotation by -yaw; (c, s) is +yaw.
        rx = c * dx + s * dy
        ry = -s * dx + c * dy
        best = min(best, point_polygon_distance(rx, ry, poly) - radius)
    return best


def moving_obstacle_radius(name):
    """Bounding circle for a moving model in limo_complex.world, in metres.

    The first ten are cross shaped pairs of cylinders; model 10 is a 5 cm
    cylinder; models 11-19 use a 0.3-scaled person mesh.
    """
    if not name.startswith('cross_clone_'):
        return None
    try:
        index = int(name.rsplit('_', 1)[1])
    except ValueError:
        return None
    if index < 10:
        return 0.26
    if index == 10:
        return 0.05
    if index < 20:
        return 0.25
    return None


def segment_clearance(p0, p1, obstacles, step=0.05):
    """Minimum distance from the segment ``p0 -> p1`` to any obstacle AABB.

    Used to reject goals the straight line cannot reach without threading past an
    obstacle: both planners under test are structurally local (cspace is a pure
    reactive policy fed the raw final-goal bearing, neupan builds a *straight*
    initial path with ``curve_style: line``), so a goal behind a concave obstacle
    measures an architectural limit rather than navigation quality.
    """
    if not obstacles:
        return float("inf")
    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0)
    n = max(int(math.ceil(length / step)), 1)
    best = float("inf")
    for i in range(n + 1):
        t = float(i) / n
        px = x0 + (x1 - x0) * t
        py = y0 + (y1 - y0) * t
        for a in obstacles:
            d = point_aabb_distance(px, py, a)
            if d < best:
                best = d
                if best <= 0.0:
                    return 0.0
    return best



def load_bench_config(path):
    """Load config/bench.yaml -- the harness parameters.  Lazy import: the module
    stays import-cheap for callers that only want the geometry helpers."""
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def world_config_path(world_path):
    """The per-world config that belongs to a world file: same directory, same
    stem, ``.yaml``.  worlds/limo_complex.world -> worlds/limo_complex.yaml.

    Derived rather than configured so the two cannot drift apart: a run cannot
    load one map's bounds and start pose while scoring another map's targets.
    """
    return os.path.splitext(os.path.abspath(world_path))[0] + ".yaml"


def load_world_config(path):
    """Load a worlds/<stem>.yaml.  Missing is an error, not an empty dict.

    An absent file would otherwise surface as a `KeyError: 'arena'` from deep
    inside the harness, where the cause (this world has no config authored yet)
    is nowhere in the message.
    """
    import yaml
    if not os.path.exists(path):
        raise ValueError(
            "no world config at %s.  Every world needs one beside it, holding "
            "`arena` (bounds, start, start_clearance) and the `goals` list; "
            "copy worlds/limo_complex.yaml as a starting point." % path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    for key in ("arena", "goals"):
        if key not in cfg:
            # No path: both callers (bench.py, run_bench.py) name the config
            # file when they print this, and repeating it reads as two files.
            raise ValueError("no `%s` section in the world config" % key)
    return cfg


def goals_from_config(cfg, obstacles=None, bounds=None, start=None):
    """Read the hand-written ``goals:`` list out of a world config.

    Targets are authored, not sampled, so this returns them in file order with
    the list index as ``i`` -- what bench.py reports per trial and what
    ``--only`` filters on.  An entry is either a mapping (``{x, y, yaw?}``) or a
    ``[x, y]`` / ``[x, y, yaw]`` sequence, because a point reads better written
    by hand as three numbers on one line.

    A point that cannot be a goal -- outside the arena, inside a static obstacle,
    or not a number at all -- raises here rather than becoming a mysterious
    failed trial.  Whether a point is *hard* is not checked: that is the
    author's call, and the whole reason the list is hand-written.

    ``distance`` is added from ``start`` when one is given.  It is reported, not
    filtered on.
    """
    raw = cfg.get("goals")
    if raw is None:
        raise ValueError(
            "no `goals:` list.  The benchmark runs one trial per entry, so it "
            "needs at least one target, e.g.\n"
            "  goals:\n    - {x: 0.0, y: 0.0, yaw: 0.0}")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("`goals:` must be a non-empty list, got %r" % (raw,))

    goals = []
    for i, entry in enumerate(raw):
        if isinstance(entry, dict):
            missing = [k for k in ("x", "y") if k not in entry]
            if missing:
                raise ValueError("goals[%d] is missing %s"
                                 % (i, " and ".join(missing)))
            x, y, yaw = entry["x"], entry["y"], entry.get("yaw", 0.0)
        elif isinstance(entry, (list, tuple)) and 2 <= len(entry) <= 3:
            x, y = entry[0], entry[1]
            yaw = entry[2] if len(entry) == 3 else 0.0
        else:
            raise ValueError("goals[%d] must be {x, y, yaw?} or [x, y, yaw?], "
                             "got %r" % (i, entry))

        for name, v in (("x", x), ("y", y), ("yaw", yaw)):
            # bool is an int in Python; `y: true` is a typo, not a coordinate.
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError("goals[%d].%s must be a number, got %r"
                                 % (i, name, v))
        x, y, yaw = float(x), float(y), float(yaw)

        if bounds is not None:
            x_min, x_max, y_min, y_max = bounds
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                raise ValueError(
                    "goals[%d] = (%.3f, %.3f) is outside arena.bounds "
                    "x[%.2f %.2f] y[%.2f %.2f]" % (i, x, y, x_min, x_max,
                                                   y_min, y_max))
        if obstacles and not check_pos(x, y, obstacles, bounds=bounds, margin=0.0):
            raise ValueError("goals[%d] = (%.3f, %.3f) is inside a static "
                             "obstacle in the world file" % (i, x, y))

        g = {"i": i, "x": x, "y": y, "yaw": yaw}
        if start is not None:
            g["distance"] = round(math.hypot(x - start[0], y - start[1]), 4)
        goals.append(g)
    return goals


def trial_plan(goals, trials=None, sample="cycle", seed=None):
    """Expand the target list into the trials to run, one target per trial.

    Returns a list of copies of the goals -- one per trial -- in the order they
    will be run, each carrying the trial number as ``i`` and the target's index
    in the config as ``goal_i``.  Every other field (x, y, yaw, distance) is the
    target's, so a caller can treat a plan entry exactly like a goal.

    `trials` defaults to one trial per target.  `sample`:
      cycle  -- shuffle the target list and walk it, reshuffling each time it is
                exhausted.  Every target is used within one of the same number
                of times, which is the point of repeating: a target that drew
                three trials against a target that drew one cannot be compared,
                and with obstacles re-randomized every trial the extra
                repetitions are the only thing that separates the two.
      random -- uniform with replacement, the literal reading of "pick one at
                random".  Allowed, and reported, but it can starve a target.

    The order is random but its *schedule* is reproducible when `seed` is given.
    The trials themselves are not: Gazebo re-randomizes the moving obstacles on
    every run, so a seed fixes which target comes when, not what happens.
    """
    if not goals:
        raise ValueError("no goals to draw trials from")
    if trials is None:
        trials = len(goals)
    if isinstance(trials, bool) or not isinstance(trials, int):
        raise ValueError("`trials:` must be a whole number, got %r" % (trials,))
    if trials < 1:
        raise ValueError("`trials:` must be at least 1, got %d" % trials)
    if sample not in ("cycle", "random"):
        raise ValueError("`goal_sample:` must be 'cycle' or 'random', got %r"
                         % (sample,))

    rng = random.Random(seed)
    order = []
    if sample == "random":
        order = [rng.randrange(len(goals)) for _ in range(trials)]
    else:
        while len(order) < trials:
            cycle = list(range(len(goals)))
            rng.shuffle(cycle)
            order.extend(cycle[:trials - len(order)])

    plan = []
    for n, k in enumerate(order):
        t = dict(goals[k])
        t["goal_i"] = k
        t["i"] = n
        plan.append(t)
    return plan


def goals_sha1(goals):
    """Canonical hash of the goal points, for report.py's comparability check.

    Hashes the normalized points rather than the config file they came from:
    reordering a comment or retuning a timeout in bench.yaml must not make two
    runs look incomparable, while moving a target must.
    """
    blob = json.dumps(goals, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------


def _selftest(world, cfg_path):
    """Assert the authored world satisfies every invariant the bench relies on.

    Runs under a bare python3 -- no master, no Gazebo -- so an authoring mistake
    (a wall too thick, a corridor too narrow, a mis-named model) fails in a
    second instead of showing up as a mysterious planner failure mid-run.

    Which invariants apply is the world's own declaration: a world config with a
    `world.moving_models` block describes a furniture-free population world and
    gets the checks that go with one.  Keying this off the filename instead would
    put the numbers in the code, where changing the world cannot reach them.
    """
    cfg = load_world_config(cfg_path) if cfg_path else {}
    arena = cfg.get("arena", {})
    declared = cfg.get("world") or {}
    if declared.get("moving_models"):
        tree = ET.parse(world)
        root = tree.getroot()
        populations = root.findall('./world/population')
        explicit = root.findall('./world/model')
        movers = [m for m in explicit if (m.get('name') or '').startswith('cross_clone_')]
        count = sum(int(p.findtext('model_count', '0')) for p in populations)
        count += len(movers)
        physics = root.find('./world/physics')
        step = float(physics.findtext('max_step_size', '0')) if physics is not None else 0
        rate = float(physics.findtext('real_time_update_rate', '0')) if physics is not None else 0
        want = int(declared["moving_models"])
        want_step = float(declared.get("physics_step", 0.01))
        want_rate = float(declared.get("physics_rate", 100))
        # Static AABBs must be exactly zero: the goal loader treats an empty
        # obstacle set as "nothing fixed to avoid", and the whole difficulty of
        # this world is supposed to come from the moving models.
        static = parse_world_obstacles(world, margin=0.0)
        ok = (count == want and step == want_step and rate == want_rate
              and not static
              and all(moving_obstacle_radius('cross_clone_%d' % i) is not None
                      for i in range(want))
              and len(arena.get('start', [])) == 3)
        print('%s: %d moving models (want %d), physics %.3f/%g (want %.3f/%g), '
              'static AABBs %d'
              % (os.path.basename(world), count, want, step, rate, want_step,
                 want_rate, len(static)))
        print('SELFTEST %s' % ('PASSED' if ok else 'FAILED'))
        return 0 if ok else 1
    fails = []

    def check(ok, msg):
        print("  [%s] %s" % ("ok  " if ok else "FAIL", msg))
        if not ok:
            fails.append(msg)

    obs_true = parse_world_obstacles(world, margin=0.0)
    print("world: %s" % world)
    print("parsed %d obstacle AABBs" % len(obs_true))
    for i, o in enumerate(obs_true):
        print("    %2d  x[%7.3f %7.3f]  y[%7.3f %7.3f]   %.2f x %.2f"
              % (i, o[0], o[1], o[2], o[3], o[1] - o[0], o[3] - o[2]))
    print()

    n_min = arena.get("min_obstacles", 1)
    check(len(obs_true) >= n_min,
          "parsed %d obstacles (>= %d)" % (len(obs_true), n_min))

    # The floor must not appear as an obstacle: its name has to contain
    # plane/ground/floor for the vendored skip list to drop it.
    tree = ET.parse(world)
    floors = [m.get('name') for m in tree.getroot().iter('model')
              if 'ground' in (m.get('name') or '') or 'plane' in (m.get('name') or '')]
    check(bool(floors), "ground plane present and skipped (name=%s)" % (floors or "NONE"))
    check(all(len(o) == 4 for o in obs_true), "every AABB is a 4-tuple")

    # No model may be nested inside another, or root.iter('model') double-counts.
    nested = [m.get('name') for m in tree.getroot().iter('model')
              if any(True for _ in m.iter('model') if _ is not m)]
    check(not nested, "no nested <model> elements%s"
          % ("" if not nested else " -- found %s" % nested))

    try:
        bounds = infer_bounds(obs_true)
        print("  inferred interior: x[%.3f %.3f] y[%.3f %.3f]"
              % (bounds[0], bounds[1], bounds[2], bounds[3]))
        want = arena.get("bounds")
        if want:
            ok = all(abs(a - b) <= 0.01 for a, b in zip(bounds, want))
            check(ok, "inferred bounds match bench.yaml %s (tol 1cm), got %s"
                  % (want, ["%.3f" % v for v in bounds]))
        else:
            check(True, "inferred bounds (no authored value to compare)")
    except ValueError as exc:
        check(False, "infer_bounds: %s" % exc)
        bounds = None

    # Every unbounded pair of obstacles must be a usable corridor.  Pairs that
    # overlap or touch are one merged obstacle (a U-shaped pocket, a wall meeting
    # the arena edge) and have no corridor between them at all.
    min_pass = arena.get("min_passage", 1.6)
    worst = None
    for i in range(len(obs_true)):
        for j in range(i + 1, len(obs_true)):
            g = aabb_gap(obs_true[i], obs_true[j])
            if g <= 1e-9:
                continue                      # merged / touching -- not a corridor
            if worst is None or g < worst[0]:
                worst = (g, i, j)
    if worst is None:
        check(True, "no free-standing obstacle pairs to check")
    else:
        check(worst[0] >= min_pass,
              "narrowest corridor %.3f m (>= %.2f) between AABB %d and %d"
              % (worst[0], min_pass, worst[1], worst[2]))

    # Start pose must be legal and have room to rotate.
    sx, sy = arena.get("start", [0.0, 0.0])[:2]
    if bounds:
        check(check_pos(sx, sy, obs_true, bounds=bounds, margin=0.0),
              "start (%.2f, %.2f) is inside the arena and clear" % (sx, sy))
        d = min(point_aabb_distance(sx, sy, o) for o in obs_true)
        check(d >= arena.get("start_clearance", 0.6),
              "start clearance %.3f m (>= %.2f)"
              % (d, arena.get("start_clearance", 0.6)))

    # Cross-check the vendored copies against the originals when ROS is around.
    sac = os.environ.get("BENCH_SAC_DIR", "/DRL-robot-navigation/SAC")
    if os.path.exists(os.path.join(sac, "laser_env.py")):
        try:
            sys.path.insert(0, sac)
            import laser_env
            ref = laser_env.parse_world_obstacles(world, margin=0.0)
            same = (len(ref) == len(obs_true)
                    and all(all(abs(a - b) < 1e-9 for a, b in zip(x, y))
                            for x, y in zip(ref, obs_true)))
            check(same, "vendored parse_world_obstacles agrees with SAC/laser_env.py")
            probe = [(0.0, 0.0), (1.0, 1.0), (-4.9, -4.9), (2.8, -2.5), (0.5, 0.0)]
            agree = all(
                laser_env.check_pos(x, y, obs_true, bounds=bounds, margin=0.2)
                == check_pos(x, y, obs_true, bounds=bounds, margin=0.2)
                for x, y in probe)
            check(agree, "vendored check_pos agrees with SAC/laser_env.py")
        except Exception as exc:                       # noqa: BLE001
            check(False, "cross-check against laser_env failed: %s" % exc)
    else:
        print("  [skip] SAC/laser_env.py not at %s -- cross-check skipped" % sac)

    print()
    if fails:
        print("SELFTEST FAILED (%d)" % len(fails))
        return 1
    print("SELFTEST PASSED")
    return 0


def footprint_report(cfg):
    """Print the configured polygons and the silhouette they imply for the laser.

    The collision test asks whether a beam came back shorter than the footprint's
    own radius in that direction, so ``d_self`` *is* the threshold, per beam.
    Printing it is the quickest way to see what a footprint change would do --
    a thin polygon has a large threshold only in a narrow forward cone, and that
    is not obvious from the vertex list.

    The body mask is printed the same way, because the two have to agree in the
    right direction: where ``d_body`` exceeds ``d_self`` the robot's own chassis
    sticks out past its scored silhouette, and the beams between the two radii
    are exactly the ones that would otherwise be read as an obstacle sitting on
    the robot's flank.
    """
    robot = cfg.get("robot") or {}
    poly = parse_footprint(robot.get("footprint"), "robot.footprint")
    lx, ly = (robot.get("laser_xyz") or (0.0, 0.0))[:2]
    length, width = polygon_extent(poly)

    print("footprint: %d point(s), extent %.3f x %.3f m"
          % (len(poly), length, width))
    print("  " + "  ".join("(%.3f, %.3f)" % p for p in poly))
    inside = point_in_polygon(lx, ly, poly)
    print("laser at (%.3f, %.3f): %s"
          % (lx, ly, "inside" if inside else "OUTSIDE -- test is meaningless"))

    body = None
    if robot.get("body"):
        body = parse_footprint(robot["body"], "robot.body")
        blen, bwid = polygon_extent(body)
        print("body mask: %d point(s), extent %.3f x %.3f m"
              % (len(body), blen, bwid))
        print("  " + "  ".join("(%.3f, %.3f)" % p for p in body))
        print("laser at (%.3f, %.3f): %s"
              % (lx, ly, "inside" if point_in_polygon(lx, ly, body)
                 else "OUTSIDE -- cannot mask its own returns"))

    print("\n  bearing   d_self   d_body   note")
    for deg, note in ((0, "straight ahead"), (8, "forward-cone edge"),
                      (45, ""), (90, "abeam"), (120, "FOV end"),
                      (180, "astern, no beam")):
        for sign in ((1,) if deg in (0, 180) else (1, -1)):
            theta = math.radians(deg * sign)
            d = polygon_ray_exit(lx, ly, theta, poly)
            b = polygon_ray_exit(lx, ly, theta, body) if body else None
            print("  %+6.1f   %s   %s   %s"
                  % (deg * sign,
                     "  none" if d is None else "%6.3f" % d,
                     "  none" if b is None else "%6.3f" % b,
                     note if sign > 0 else ""))
    return 0


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    default_world = os.path.join(here, "worlds", "limo_complex.world")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--world", default=default_world)
    ap.add_argument("--config", default=None,
                    help="world config; default: <world stem>.yaml beside the "
                         "world file")
    ap.add_argument("--bench-config",
                    default=os.path.join(here, "config", "bench.yaml"),
                    help="harness config holding the footprint; --footprint "
                         "reads it")
    ap.add_argument("--footprint", action="store_true",
                    help="print the configured footprint polygon and the laser "
                         "silhouette distance it implies per bearing")
    args = ap.parse_args(argv)

    if args.footprint:
        return footprint_report(load_bench_config(args.bench_config))

    if args.selftest:
        return _selftest(args.world, args.config or world_config_path(args.world))

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
