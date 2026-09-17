#!/usr/bin/env python3

"""Generate the benchmark goal set -- once, seeded, and committed.

Both planners are evaluated against the SAME file, so this runs before either
run and its output is a reviewable artefact rather than a side effect.  The
whole point is that `planner:=cspace` and `planner:=neupan` differ in exactly
one token; if each run picked its own goals the comparison would be meaningless.

  rosrun planner_bench gen_goals.py --n 12 --seed 0
  rosrun planner_bench gen_goals.py --check          # byte-identical reproduction?
  rosrun planner_bench gen_goals.py --selfcheck      # write the negative-control set

Rejection rules, and why each exists:

  * inside an obstacle, or within `goal.clearance` of one -- sized at 0.8 m for
    the MOST conservative planner under test.  cspace dilates obstacles by its
    1.20 m training footprint and stops at 0.3 m, so a goal inside that ring is
    one it can never reach; scoring that as a cspace failure would hand neupan a
    free point.
  * outside the arena.
  * too close to the start, or closer than `goal.separation` to another goal.
  * distance outside [min_distance, max_distance].  cspace trained on targets
    2.4-3.6 m out and truncated episodes beyond 4 m, so a longer goal is out of
    distribution for it rather than merely harder.
  * **no line of sight** -- the important one.  Both planners are structurally
    local: cspace is a reactive policy fed the raw final-goal bearing, and neupan
    builds a *straight* initial path (`curve_style: line`).  A goal behind a
    concave obstacle is unsolvable for both by construction, so testing one would
    measure an architectural limit instead of navigation quality.  The dead-end
    pocket stays in the world as a hazard a bad policy can fall into; no goal is
    ever placed behind it.
"""

import argparse
import json
import math
import os
import random
import sys

# installed, this script sits next to bench_common.py in lib/<pkg>/; run from the
# source tree, the module is one directory over in src/
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, os.pardir, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench_common import (check_pos, load_bench_config, parse_world_obstacles,
                          segment_clearance, sha1_of)


def band_of(los_clearance, bands):
    if los_clearance <= bands["tight"]:
        return "tight"
    if los_clearance <= bands["medium"]:
        return "medium"
    return "open"


def sample_candidates(start, obstacles, bounds, cfg, rng, want, max_tries):
    """Collect rejected-filtered candidates, bucketed by difficulty band."""
    g = cfg["goal"]
    sx, sy = start[0], start[1]
    buckets = {"tight": [], "medium": [], "open": []}
    n_reject = {"obstacle": 0, "los": 0, "distance": 0, "start": 0}

    for _ in range(max_tries):
        # Keep a deep pool: `select` discards candidates that violate the mutual
        # separation, and with only a few hundred candidates it can run dry
        # before reaching --n even though legal goals still exist.
        if sum(len(v) for v in buckets.values()) >= want * 200:
            break
        r = rng.uniform(g["min_distance"], g["max_distance"])
        t = rng.uniform(-math.pi, math.pi)
        gx = sx + r * math.cos(t)
        gy = sy + r * math.sin(t)

        d = math.hypot(gx - sx, gy - sy)
        if d < g["min_from_start"]:
            n_reject["start"] += 1
            continue
        if not (g["min_distance"] <= d <= g["max_distance"]):
            n_reject["distance"] += 1
            continue
        if not check_pos(gx, gy, obstacles, bounds=bounds, margin=g["clearance"]):
            n_reject["obstacle"] += 1
            continue

        los = segment_clearance((sx, sy), (gx, gy), obstacles)
        if los < g["los_margin"]:
            n_reject["los"] += 1
            continue

        buckets[band_of(los, cfg["goal"]["bands"])].append({
            "x": round(gx, 4),
            "y": round(gy, 4),
            "distance": round(d, 4),
            "los_clearance": round(los, 4),
            "band": band_of(los, cfg["goal"]["bands"]),
        })

    return buckets, n_reject


def select(buckets, want, separation, rng):
    """Round-robin the difficulty bands, enforcing minimum mutual separation.

    Round-robin rather than "take the first N" so the set is not dominated by
    whichever band happens to be easiest to sample -- with N ~ 12 an unbalanced
    set makes the per-band breakdown useless.
    """
    for b in buckets.values():
        rng.shuffle(b)

    picked = []
    order = ["tight", "medium", "open"]
    while len(picked) < want:
        progressed = False
        for name in order:
            if len(picked) >= want:
                break
            bucket = buckets[name]
            while bucket:
                cand = bucket.pop(0)
                if all(math.hypot(cand["x"] - p["x"], cand["y"] - p["y"]) >= separation
                       for p in picked):
                    picked.append(cand)
                    progressed = True
                    break
        if not progressed:
            break        # every remaining candidate is too close to a picked one
    return picked


def build(args, cfg, obstacles, bounds):
    start = list(args.start)
    rng = random.Random(args.seed)

    buckets, n_reject = sample_candidates(
        start, obstacles, bounds, cfg, rng, args.n, args.max_tries)
    picked = select(buckets, args.n, cfg["goal"]["separation"], rng)

    goals = []
    for i, cand in enumerate(picked):
        # yaw = the travel direction from the start.  Stored so the goal set is
        # self-describing, but note it is a neupan-only field: cspace never reads
        # PoseStamped.orientation (or even header.frame_id), so a yaw would tilt
        # the comparison one way for free.  Success is position-only for both.
        yaw = math.atan2(cand["y"] - start[1], cand["x"] - start[0])
        goals.append({
            "i": i,
            "x": cand["x"],
            "y": cand["y"],
            "yaw": round(yaw, 6),
            "distance": cand["distance"],
            "los_clearance": cand["los_clearance"],
            "band": cand["band"],
        })

    return {
        "meta": {
            "world": world_ref(args.world),
            "world_sha1": sha1_of(args.world),
            "seed": args.seed,
            "start": start,
            "bounds": list(bounds),
            "n": len(goals),
            "generator": "planner_bench/scripts/gen_goals.py",
            "yaw_used_by": "neupan_only",
            "bands": cfg["goal"]["bands"],
            "rejected": n_reject,
            "goal_thresholds": cfg["goal"],
            "robot": cfg["robot"],
            "trial": cfg["trial"],
        },
        "goals": goals,
    }


_PKG = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


def world_ref(path):
    """The world path as recorded in the goal set.

    Relative to the package when the world is inside it.  An absolute path makes
    the file unreproducible across environments for no reason: the same repo at
    a different prefix -- or bind-mounted at /DRL-robot-navigation inside the
    container -- regenerates different bytes, and `--check` then reports "not
    reproducible from seed 0" when the goals are in fact identical and only the
    path prefix moved.  A reproducibility check that cries wolf is worse than
    none, because it teaches you to ignore it.
    """
    ap = os.path.abspath(path)
    if ap == _PKG or ap.startswith(_PKG + os.sep):
        return os.path.relpath(ap, _PKG)
    return ap


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.normpath(os.path.join(here, os.pardir))

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default=os.path.join(pkg, "worlds", "bench_arena.world"))
    ap.add_argument("--config", default=os.path.join(pkg, "config", "bench.yaml"))
    # Resolved after parsing, because --selfcheck must not land on the committed
    # goal set: it bypasses the overwrite guard below (it is meant to be
    # re-runnable), so a plain `gen_goals.py --selfcheck` -- which is what the
    # module docstring tells you to type -- would silently replace the benchmark's
    # goal set with the three-goal negative control and void every comparison.
    ap.add_argument("--out", default=None, help="default: "
                    "config/goal_sets/bench_seed0.json, or negcontrol.json "
                    "when --selfcheck is given")
    # 11 is the packing ceiling for this arena at the configured clearance, not
    # an arbitrary choice: every goal needs >= goal.clearance (0.8 m) from every
    # obstacle AND >= goal.separation (1.0 m) from every other goal, all inside
    # an annulus of goal.min_distance..max_distance around one start pose.
    # Asking for more is not an error -- it just yields fewer and warns.
    ap.add_argument("--n", type=int, default=11)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "YAW"),
                    help="default: arena.start from bench.yaml")
    ap.add_argument("--max-tries", type=int, default=200000)
    ap.add_argument("--check", action="store_true",
                    help="regenerate and assert byte-identical to --out")
    ap.add_argument("--selfcheck", action="store_true",
                    help="write the negative-control set instead (see below)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing goal set")
    args = ap.parse_args(argv)

    if args.out is None:
        args.out = os.path.join(pkg, "config", "goal_sets",
                                "negcontrol.json" if args.selfcheck
                                else "bench_seed0.json")

    cfg = load_bench_config(args.config)
    if args.start is None:
        args.start = cfg["arena"]["start"]

    obstacles = parse_world_obstacles(args.world, margin=0.0)
    bounds = tuple(cfg["arena"]["bounds"])

    # The self-check set is the harness's own falsification test: one ordinary
    # reachable goal plus one deliberately placed on the far side of the divider,
    # which has NO line of sight from the start.  A correct harness must score
    # the second as timeout or stuck -- if it reports success for it, the success
    # criterion is vacuous and neither planner's numbers mean anything.
    if args.selfcheck:
        sx, sy = args.start[0], args.start[1]
        # The reachable control is TAKEN FROM THE COMMITTED BENCHMARK SET, not
        # hand-written.  A hand-placed goal is exactly as good as the author's
        # intuition about reachability, and this one was wrong: the first
        # version put it 4.6 m out, beyond the 3.6 m the generator's own
        # max_distance exists to enforce, and cspace -- which truncates episodes
        # past 4 m -- stalled on it and on both unreachable controls.  The
        # control then said "the harness cannot pass anything", which was a
        # statement about the goal, not the harness.
        #
        # A goal both planners already reached under the benchmark rules is
        # reachable by demonstration.  The most open one is chosen so the guard
        # is as strong as possible.
        bench_path = os.path.join(pkg, "config", "goal_sets", "bench_seed0.json")
        try:
            with open(bench_path) as fh:
                bench_goals = json.load(fh)["goals"]
        except (IOError, KeyError, ValueError) as exc:
            print("FAIL: --selfcheck needs %s to borrow a proven-reachable goal "
                  "(%s).  Generate it first." % (bench_path, exc))
            return 1
        proven = max(bench_goals, key=lambda g: g["los_clearance"])

        # Three controls, ordered by how airtight the expectation is.
        #
        # The third one is the load-bearing one.  "Behind the divider" is a
        # *soft* control: a success there would be ambiguous, because it could
        # mean the criterion is vacuous OR that the planner genuinely found the
        # 2 m doorway and drove around -- the harness cannot tell those apart.
        # A goal beyond the boundary wall cannot be excused that way: the robot
        # physically cannot leave the arena, so the closest it can ever come is
        # 2.0 m against a 0.5 m success_radius.  A success there is proof.
        #
        # Controls 1 and 2 deliberately sit outside the generator's distance
        # band -- nothing inside a 3.6 m radius of the start is both walled off
        # and 0.8 m clear of the divider.  That is fine for a "must not
        # succeed" assertion: being out of distribution can only make a planner
        # fail harder, never succeed spuriously.  It is NOT fine for the
        # "must succeed" guard, which is why that one is borrowed.
        controls = [
            (proven["x"], proven["y"],
             "borrowed from bench_seed0.json (los %.2f m): both planners "
             "reached it, so the harness MUST score it success"
             % proven["los_clearance"]),
            (3.6, -3.6,
             "behind the divider, no line of sight -- a local planner should fail"),
            (7.0, 0.0,
             "1.75 m OUTSIDE the east wall: unreachable by construction, so "
             "success here is proof the criterion is vacuous"),
        ]
        goals = []
        for i, (gx, gy, why) in enumerate(controls):
            g = {"i": i, "x": gx, "y": gy, "yaw": 0.0, "band": "control",
                 "why": why}
            g["distance"] = round(math.hypot(gx - sx, gy - sy), 4)
            g["los_clearance"] = round(
                segment_clearance((sx, sy), (gx, gy), obstacles), 4)
            goals.append(g)
        doc = {
            "meta": {
                "world": world_ref(args.world),
                "world_sha1": sha1_of(args.world),
                "seed": args.seed, "start": list(args.start),
                "bounds": list(bounds), "n": len(goals),
                "generator": "planner_bench/scripts/gen_goals.py --selfcheck",
                "note": "negative control. Goal 0 is reachable and MUST be a "
                        "success; goals 1 (walled off) and 2 (outside the "
                        "arena) MUST NOT be. A success on goal 2 in "
                        "particular proves the success criterion is vacuous.",
                "goal_thresholds": cfg["goal"], "robot": cfg["robot"],
                "trial": cfg["trial"],
            },
            "goals": goals,
        }
        for g, (_x, _y, why) in zip(goals, controls):
            print("goal %d  los=%.2f m  %s" % (g["i"], g["los_clearance"], why))
    else:
        doc = build(args, cfg, obstacles, bounds)

    text = json.dumps(doc, indent=2, sort_keys=True) + "\n"

    if args.check:
        try:
            with open(args.out) as fh:
                have = fh.read()
        except IOError:
            print("FAIL: %s does not exist -- generate it first" % args.out)
            return 1
        if have != text:
            print("FAIL: %s is not reproducible from seed %d" % (args.out, args.seed))
            return 1
        print("OK: %s reproduces byte-for-byte (seed %d, %d goals)"
              % (args.out, args.seed, doc["meta"]["n"]))
        return 0

    if os.path.exists(args.out) and not args.force and not args.selfcheck:
        print("refusing to overwrite %s without --force" % args.out)
        return 1

    if not args.selfcheck:
        print("world : %s" % os.path.basename(args.world))
        print("start : (%.2f, %.2f, %.2f)" % tuple(args.start))
        print("bounds: x[%.1f %.1f] y[%.1f %.1f]" % bounds)
        print("rejected: %s" % doc["meta"]["rejected"])
        print()
        print("  i     x       y     dist   los   band")
        for g in doc["goals"]:
            print("  %2d  %6.2f  %6.2f  %5.2f  %5.2f  %s"
                  % (g["i"], g["x"], g["y"], g["distance"],
                     g["los_clearance"], g["band"]))
        counts = {}
        for g in doc["goals"]:
            counts[g["band"]] = counts.get(g["band"], 0) + 1
        print("\nbands: %s" % counts)
        if doc["meta"]["n"] < args.n:
            print("WARNING: wanted %d goals, produced %d (raise --max-tries or "
                  "loosen a threshold)" % (args.n, doc["meta"]["n"]))

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(args.out, 'w') as fh:
        fh.write(text)
    print("\nwrote %s (%d goals)" % (args.out, doc["meta"]["n"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
