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


def obb_aabb_clearance(cx, cy, yaw, half_len, half_wid, aabb):
    """Signed clearance between a yaw-rotated rectangle and an axis-aligned box.

    Negative means penetration, and its magnitude is the penetration depth;
    positive is the true Euclidean separation.  Exact for two convex polygons:
    the separating-axis theorem says the distance is the largest projection gap
    over the face normals of both shapes, which for a rectangle is its two body
    axes plus the AABB's two world axes.
    """
    c = math.cos(yaw)
    s = math.sin(yaw)
    axes = ((1.0, 0.0), (0.0, 1.0), (c, s), (-s, c))

    corners = []
    for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        px = sx * half_len
        py = sy * half_wid
        corners.append((cx + px * c - py * s, cy + px * s + py * c))
    box = ((aabb[0], aabb[2]), (aabb[1], aabb[2]),
           (aabb[1], aabb[3]), (aabb[0], aabb[3]))

    gaps = []
    for nx, ny in axes:
        ra = [p[0] * nx + p[1] * ny for p in corners]
        rb = [p[0] * nx + p[1] * ny for p in box]
        # > 0 when the projections are disjoint, and then the gap is the distance
        # along this axis; <= 0 when they overlap.
        gaps.append(max(min(rb) - max(ra), min(ra) - max(rb)))

    separated = [g for g in gaps if g > 0.0]
    if separated:
        return max(separated)
    # Overlapping on every axis: the shallowest push-out is the penetration depth.
    return max(gaps)


def min_clearance(cx, cy, yaw, half_len, half_wid, obstacles):
    """Signed clearance from the robot rectangle to the nearest obstacle AABB."""
    if not obstacles:
        return float("inf")
    return min(obb_aabb_clearance(cx, cy, yaw, half_len, half_wid, a)
               for a in obstacles)


def moving_obstacle_radius(name):
    """Bounding circle for a moving model in limo_complex.world, in metres.

    The first ten are cross shaped pairs of cylinders; model 10 is a 5 cm
    cylinder; models 11-19 use a 0.3-scaled person mesh. Mesh clearance is an
    approximation and is reported as such in the benchmark documentation.
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


def moving_clearance(cx, cy, yaw, half_len, half_wid, models):
    """Signed clearance from Limo rectangle to moving bounding circles.

    models is an iterable of (name, x, y) ground-truth Gazebo poses.
    """
    best = float('inf')
    c, s = math.cos(yaw), math.sin(yaw)
    for name, x, y in models:
        radius = moving_obstacle_radius(name)
        if radius is None:
            continue
        dx, dy = x - cx, y - cy
        rx, ry = c * dx + s * dy, -s * dx + c * dy
        ox = max(abs(rx) - half_len, 0.0)
        oy = max(abs(ry) - half_wid, 0.0)
        best = min(best, math.hypot(ox, oy) - radius)
    return best


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


def _rot_rpy(rpy):
    """URDF rpy (roll, pitch, yaw) -> 3x3 rotation matrix as a tuple of rows.

    URDF convention is R = Rz(yaw) * Ry(pitch) * Rx(roll).
    """
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp,     cp * sr,                cp * cr),
    )


def _mat_mul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))


def _mat_vec(a, v):
    return tuple(sum(a[i][k] * v[k] for k in range(3)) for i in range(3))


def _primitive_points(geom):
    """Sample a URDF collision primitive's extreme 3D points (local frame).

    Sampling rather than solving for the AABB analytically is what makes a
    rotated cylinder come out right: a Limo wheel is ``<cylinder length=.045
    radius=.045>`` under ``rpy="1.57 0 0"``, so its axis lies along Y and its XY
    footprint is a 0.09 x 0.045 *rectangle*.  Treating it as a bounding square
    (the convention parse_world_obstacles uses for SDF, and the convention
    training used) would inflate the robot's half-width from 0.11 to 0.155.

    Returns (points, inflate) where `inflate` is an isotropic pad to apply after
    projecting -- a sphere projects to a point plus its radius.
    """
    box = geom.find('box')
    cyl = geom.find('cylinder')
    sph = geom.find('sphere')
    if box is not None:
        sx, sy, sz = [float(v) for v in box.get('size', '0 0 0').split()]
        return ([(sx / 2.0 * i, sy / 2.0 * j, sz / 2.0 * k)
                 for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], 0.0)
    if cyl is not None:
        r = float(cyl.get('radius', '0'))
        L = float(cyl.get('length', '0'))
        pts = []
        for k in (-1, 1):
            for i in range(24):
                t = 2.0 * math.pi * i / 24.0
                pts.append((r * math.cos(t), r * math.sin(t), k * L / 2.0))
        return (pts, 0.0)
    if sph is not None:
        return ([(0.0, 0.0, 0.0)], float(sph.get('radius', '0')))
    return (None, 0.0)


def footprint_from_urdf(xml_string, root_link="base_footprint"):
    """Derive the robot's (half_length, half_width) from a URDF string.

    Composes every joint at its zero configuration -- exact for fixed joints, and
    for revolute/continuous ones at rest (a wheel's spin does not move its XY
    footprint) -- then unions every collision primitive's projected XY extent
    into one AABB in the ``root_link`` frame.

    Derived rather than hardcoded so the collision ground truth matches Gazebo's
    own collision primitives, which is what makes ``clearance <= 0`` mean "the
    robot is actually touching something" rather than "someone's guess about the
    robot says so".  ``urdf_parser_py`` is not installed in this image, hence the
    stdlib walk; note URDF puts geometry in *attributes* (``<box size="..."/>``)
    where SDF uses child elements, which is why this cannot share code with
    ``parse_world_obstacles``.

    Returns ``(half_length, half_width, n_collisions)``.
    """
    root = ET.fromstring(xml_string)

    def _floats(text, n):
        vals = [float(v) for v in (text or '').split()]
        while len(vals) < n:
            vals.append(0.0)
        return vals[:n]

    def _origin(elem):
        o = elem.find('origin')
        if o is None:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        return _floats(o.get('xyz'), 3), _floats(o.get('rpy'), 3)

    # link -> (parent, xyz, rpy) for every joint, whatever its type.
    joints = {}
    for j in root.findall('joint'):
        parent = j.find('parent')
        child = j.find('child')
        if parent is None or child is None:
            continue
        xyz, rpy = _origin(j)
        joints[child.get('link')] = (parent.get('link'), xyz, rpy)

    resolved = {}

    def resolve(link, seen):
        """-> (R, t) taking points in `link` into the root link's frame."""
        if link in resolved:
            return resolved[link]
        if link in seen:
            raise ValueError("joint cycle through link '%s'" % link)
        if link not in joints:
            resolved[link] = (((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)), (0.0, 0.0, 0.0))
            return resolved[link]
        parent, xyz, rpy = joints[link]
        pr, pt = resolve(parent, seen | {link})
        cr = _rot_rpy(rpy)
        resolved[link] = (_mat_mul(pr, cr),
                          tuple(pt[i] + _mat_vec(pr, xyz)[i] for i in range(3)))
        return resolved[link]

    xs, ys, count = [], [], 0
    for link in root.findall('link'):
        name = link.get('name')
        if name not in joints and name != root_link:
            continue                      # not part of the root's tree
        lr, lt = resolve(name, frozenset())

        for coll in link.findall('collision'):
            geom = coll.find('geometry')
            if geom is None:
                continue
            pts, inflate = _primitive_points(geom)
            if pts is None:
                continue

            xyz, rpy = _origin(coll)
            cr = _rot_rpy(rpy)
            # collision origin in root frame, then each sampled point in root frame
            base = tuple(lt[i] + _mat_vec(lr, xyz)[i] for i in range(3))
            m = _mat_mul(lr, cr)
            for pt in pts:
                w = _mat_vec(m, pt)
                xs.append(base[0] + w[0])
                ys.append(base[1] + w[1])
            if inflate:
                xs.extend([base[0] - inflate, base[0] + inflate])
                ys.extend([base[1] - inflate, base[1] + inflate])
            count += 1

    if not xs:
        raise ValueError("no collision geometry found in the URDF")

    return (max(abs(min(xs)), abs(max(xs))), max(abs(min(ys)), abs(max(ys))), count)


def load_bench_config(path):
    """Load bench.yaml.  Imported lazily so the module stays import-cheap."""
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def goals_from_config(cfg, obstacles=None, bounds=None, start=None):
    """Read the hand-written ``goals:`` list out of bench.yaml.

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
            "bench.yaml has no `goals:` list.  The benchmark runs one trial per "
            "entry, so it needs at least one target, e.g.\n"
            "  goals:\n    - {x: 0.0, y: 0.0, yaw: 0.0}")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("bench.yaml `goals:` must be a non-empty list, got %r"
                         % (raw,))

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
    """
    cfg = load_bench_config(cfg_path) if cfg_path else {}
    arena = cfg.get("arena", {})
    if os.path.basename(world) == 'limo_complex.world':
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
        ok = (count == 20 and step == 0.01 and rate == 100
              and not parse_world_obstacles(world, margin=0.0)
              and all(moving_obstacle_radius('cross_clone_%d' % i) is not None
                      for i in range(20))
              and len(arena.get('start', [])) == 3)
        print('limo_complex: %d moving models, physics %.3f/%g, static AABBs %d'
              % (count, step, rate, len(parse_world_obstacles(world, 0.0))))
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


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    default_world = os.path.join(here, "worlds", "limo_complex.world")
    default_cfg = os.path.join(here, "config", "bench.yaml")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--world", default=default_world)
    ap.add_argument("--config", default=default_cfg)
    ap.add_argument("--urdf", default=None,
                    help="also derive the footprint from this URDF/xacro output")
    args = ap.parse_args(argv)

    if args.urdf:
        with open(args.urdf) as fh:
            hl, hw, n = footprint_from_urdf(fh.read())
        print("footprint from %s: half_length=%.4f half_width=%.4f (%d collisions)"
              % (args.urdf, hl, hw, n))
        print("  -> %.4f x %.4f m" % (2 * hl, 2 * hw))
        return 0

    if args.selftest:
        return _selftest(args.world, args.config)

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
