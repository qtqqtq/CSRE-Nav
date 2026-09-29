#!/usr/bin/env python3

"""ROS-free geometry and configuration helpers for the benchmark scripts."""

import argparse
import hashlib
import json
import math
import os
import random
import sys


def sha1_of(path):
    """Content hash of a file, used to pin the world a run was scored against.

    Lives here rather than in each script so the goal set, the harness and the
    report all mean the same thing by "the world hash".
    """
    with open(path, 'rb') as fh:
        return hashlib.sha1(fh.read()).hexdigest()


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


def goals_from_config(cfg, bounds=None, start=None):
    """Read the hand-written ``goals:`` list out of a world config.

    Targets are authored, not sampled, so this returns them in file order with
    the list index as ``i`` -- what bench.py reports per trial and what
    ``--only`` filters on.  An entry is either a mapping (``{x, y, yaw?}``) or a
    ``[x, y]`` / ``[x, y, yaw]`` sequence, because a point reads better written
    by hand as three numbers on one line.

    A point outside the declared arena or one that is not numeric raises here.
    Whether a point is reachable is left to the authored benchmark scenario.

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

    ap = argparse.ArgumentParser(description=__doc__)
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

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
