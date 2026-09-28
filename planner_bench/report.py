#!/usr/bin/env python3

"""Compare two planner_bench runs side by side.

    python3 report.py reports/cspace_run.json reports/neupan_run.json
    python3 report.py                      # both files in reports/

Pure stdlib and no ROS on purpose: this reads two JSON reports and prints a
table, and it has to keep working after the simulation is gone.

Two rules the report follows, because both change what the numbers mean:

  * It refuses to compare runs whose world or target set differ.  A success rate
    against a different set of goals is not a comparison, it is two unrelated
    measurements that happen to be printed next to each other.

  * Time to goal is reported as median and IQR, never mean.  It is right
    censored -- a planner that never arrives has no arrival time, and dropping
    those trials while averaging the rest flatters whichever planner failed
    more.  The denominator is printed so the censoring is visible.

  * Results are grouped by TARGET, not by trial.  Each trial draws a target at
    random, so trial 3 of one run and trial 3 of the other are unrelated
    episodes against unrelated targets; pairing them would be an invention.
    Trials of the same target are not paired either -- the moving obstacles are
    re-randomized for every trial -- so a per-target row compares distributions,
    and the count of trials behind each row is printed to keep that honest.
"""

import argparse
import json
import math
import os
import sys


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion.

    Not the normal approximation: with a handful of trials and rates near 0 or 1
    that interval runs outside [0, 1] and reads as precision the run does not
    have.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = float(k) / n
    d = 1.0 + z * z / n
    c = p + z * z / (2.0 * n)
    r = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return ((c - r) / d, (c + r) / d)


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def med(xs):
    return quantile(xs, 0.5)


def fmt(x, spec="%.2f", dash="-"):
    if x is None or x != x:
        return dash
    return spec % x


class Run(object):
    def __init__(self, path):
        with open(path) as fh:
            self.doc = json.load(fh)
        self.path = path
        self.name = self.doc["meta"]["planner"]
        self.trials = self.doc["trials"]
        # Group by the target each trial drew, which is the only meaningful
        # grouping once one target can be drawn by several trials.  Reports
        # written before trials were sampled have no goal_i and ran one target
        # per trial, where i == goal_i, so falling back to i groups them right.
        self.by_goal = {}
        for t in self.trials:
            self.by_goal.setdefault(t.get("goal_i", t["i"]), []).append(t)

    def count(self, verdict):
        return sum(1 for t in self.trials if t["verdict"] == verdict)

    def target(self, gi):
        """The point behind a target index, taken from the trials that ran it."""
        t = self.by_goal[gi][0]
        g = t.get("goal") or {}
        return {"i": gi, "x": g.get("x"), "y": g.get("y"),
                "yaw": g.get("yaw", 0.0), "distance": g.get("distance")}

    def success_rate(self, gi):
        ts = self.by_goal[gi]
        k = sum(1 for t in ts if t["verdict"] == "success")
        return k, len(ts)

    @property
    def n(self):
        return len(self.trials)

    def median_of(self, key, only_verdict=None):
        vals = [t[key] for t in self.trials
                if t.get(key) is not None
                and (only_verdict is None or t["verdict"] == only_verdict)]
        return med(vals) if vals else None

    def iqr_of(self, key, only_verdict=None):
        vals = [t[key] for t in self.trials
                if t.get(key) is not None
                and (only_verdict is None or t["verdict"] == only_verdict)]
        if len(vals) < 2:
            return None
        return (quantile(vals, 0.25), quantile(vals, 0.75))

    def extreme(self, key, pick=min):
        vals = [t[key] for t in self.trials if t.get(key) is not None]
        return pick(vals) if vals else None

    def achieved_speed(self, t):
        """The robot's real mean speed for one trial.

        Reports written before mean_speed was split recorded the COMMANDED
        /cmd_vel average under that name, which reads as a healthy 0.18 m/s for
        a planner whose robot never moved.  Those files are still valid data, so
        recover the achieved speed from the ground-truth trace they carry rather
        than either trusting the mislabelled field or discarding the run.
        """
        if "mean_cmd_speed" in t:
            return t.get("mean_speed")
        log = t.get("log") or []
        total = span = 0.0
        for a, b in zip(log, log[1:]):
            dt = b["t"] - a["t"]
            if dt <= 0:
                continue
            total += math.hypot(b["x"] - a["x"], b["y"] - a["y"])
            span += dt
        return total / span if span > 0 else None

    def commanded_speed(self, t):
        if "mean_cmd_speed" in t:
            return t.get("mean_cmd_speed")
        return t.get("mean_speed")


VERDICTS = ["success", "collision", "stuck", "timeout", "no_motion",
            "no_planner", "aborted"]


def check_comparable(a, b):
    """Refuse to print a comparison that would not mean anything."""
    problems = []
    wa, wb = a.doc["meta"].get("world_sha1"), b.doc["meta"].get("world_sha1")
    if wa and wb and wa != wb:
        # Name the hashes, not just the basenames.  The two runs normally share
        # a path -- that is the point of a fixed world -- so printing two
        # identical filenames tells the reader nothing about what went wrong.
        # The hashes are the thing that actually differs, and the usual cause is
        # that the world file was edited between the two runs.
        problems.append("different worlds -- same path, different content "
                        "(%s: %s vs %s: %s)"
                        % (a.name, wa[:12], b.name, wb[:12]))
    # Only the points are compared, not the file they were written in: the
    # targets live in a config beside the trial timeouts, and retuning one of
    # those must not make two runs over identical targets look incomparable.
    ha = a.doc["meta"].get("goals_sha1")
    hb = b.doc["meta"].get("goals_sha1")
    if ha and hb and ha != hb:
        problems.append("different goal points (%s vs %s)" % (ha[:12], hb[:12]))
    sa = a.doc["meta"].get("start")
    sb = b.doc["meta"].get("start")
    if sa and sb and any(abs(x - y) > 1e-6 for x, y in zip(sa, sb)):
        problems.append("different start poses (%s vs %s)" % (sa, sb))
    # The targets actually exercised, not the trials run.  The trial numbering
    # is a schedule -- which target happened to come up first -- and two runs of
    # the same targets in a different order are the same experiment.  A run that
    # only touched some of the targets (`--only-goal`) is a different one, and
    # its per-target rows would sit beside rows the other run never measured.
    ga, gb = set(a.by_goal), set(b.by_goal)
    if ga != gb:
        problems.append("different targets exercised (%s vs %s)"
                        % (sorted(ga), sorted(gb)))
    return problems


def balance_note(run):
    """Trials per target, for the reader to judge how thin a per-target row is.

    Not a comparability problem: both runs come from the same `trials:` count,
    so a mismatch here is the random draw, not a different experiment.  It is
    printed because a target scored 2/2 and one scored 2/4 read very differently
    and the table alone does not say which is which across the two runs.
    """
    return [len(run.by_goal[gi]) for gi in sorted(run.by_goal)]


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reports", nargs="*",
                    help="two planner run JSONs; default: every *_run.json in "
                         "reports/")
    ap.add_argument("--out", default=None, help="also write the table to a file")
    ap.add_argument("--force", action="store_true",
                    help="print the comparison even if the runs are not comparable")
    args = ap.parse_args(argv)

    paths = args.reports
    if not paths:
        here = os.path.dirname(os.path.abspath(__file__))
        rdir = os.path.join(here, "reports")
        if os.path.isdir(rdir):
            paths = sorted(os.path.join(rdir, f) for f in os.listdir(rdir)
                           if f.endswith("_run.json"))
        if len(paths) != 2:
            sys.exit("expected exactly 2 reports in %s, found %d.  Run both "
                     "planners first:\n"
                     "  ./planner_bench/run_bench.py both"
                     % (rdir, len(paths)))
    if len(paths) != 2:
        sys.exit("expected exactly 2 reports, got %d" % len(paths))

    a, b = Run(paths[0]), Run(paths[1])
    problems = check_comparable(a, b)
    out = []
    w = out.append

    w("planner_bench comparison")
    w("=" * 68)
    meta = a.doc["meta"]
    w("world     : %s" % os.path.basename(meta["world"]))
    w("            sha1 %s" % (meta.get("world_sha1") or "-")[:16])
    w("targets   : %d point(s) from %s"
      % (len(a.by_goal), os.path.basename(meta.get("world_config", "-"))))
    w("            sha1 %s" % (meta.get("goals_sha1") or "-")[:16])
    w("draw      : %s, seed %s"
      % (meta.get("goal_sample", "one trial per target"),
         meta.get("trial_seed") if meta.get("trial_seed") is not None else "none"))
    w("start     : %s" % (meta.get("start"),))
    w("robot     : %.3f x %.3f m (%s) / %.3f x %.3f m (%s)"
      % (2 * a.doc["meta"]["robot"]["half_length"],
         2 * a.doc["meta"]["robot"]["half_width"], a.name,
         2 * b.doc["meta"]["robot"]["half_length"],
         2 * b.doc["meta"]["robot"]["half_width"], b.name))
    w("")

    if problems and not args.force:
        w("NOT COMPARABLE:")
        for p in problems:
            w("  * %s" % p)
        w("")
        w("These runs do not share a world, a target set, a start pose and the "
          "targets they")
        w("exercised, so a difference between them cannot be attributed to the "
          "planners.")
        w("Re-run both, or pass --force to print anyway.")
        text = "\n".join(out)
        print(text)
        if args.out:
            open(args.out, "w").write(text + "\n")
        return 1
    for p in problems:
        w("WARNING: %s" % p)
    if problems:
        w("")

    # ---------------------------------------------------------------- headline
    w("%-26s %-18s %-18s" % ("", a.name, b.name))
    w("-" * 68)
    w("%-26s %-18s %-18s" % ("trials", a.n, b.n))
    w("%-26s %-18s %-18s" % ("targets", len(a.by_goal), len(b.by_goal)))

    for v in VERDICTS:
        ca, cb = a.count(v), b.count(v)
        if ca == 0 and cb == 0:
            continue
        w("%-26s %-18s %-18s" % (
            v, "%d  (%.1f%%)" % (ca, 100.0 * ca / a.n),
            "%d  (%.1f%%)" % (cb, 100.0 * cb / b.n)))

    w("-" * 68)
    ci = []
    for run in (a, b):
        lo, hi = wilson(run.count("success"), run.n)
        ci.append("%.1f%%  [%.1f, %.1f]" % (
            100.0 * run.count("success") / run.n, 100 * lo, 100 * hi))
    w("%-26s %-18s %-18s" % ("success rate", ci[0], ci[1]))
    w("%-26s %-18s %-18s" % ("  95% Wilson CI", "on %d trials" % a.n,
                             "on %d trials" % b.n))
    w("")

    # Censored: only successful trials have an arrival time, so the denominator
    # is printed rather than implied.
    w("%-26s %-18s %-18s" % ("time to goal (s)", a.name, b.name))
    la = "%s  (n=%d)" % (fmt(a.median_of("duration", "success"), "%.1f"),
                         a.count("success"))
    lb = "%s  (n=%d)" % (fmt(b.median_of("duration", "success"), "%.1f"),
                         b.count("success"))
    w("%-26s %-18s %-18s" % ("median", la, lb))
    ia, ib = a.iqr_of("duration", "success"), b.iqr_of("duration", "success")
    w("%-26s %-18s %-18s" % (
        "  IQR",
        "%s" % (fmt(ia[0], "%.1f") + " - " + fmt(ia[1], "%.1f") if ia else "-"),
        "%s" % (fmt(ib[0], "%.1f") + " - " + fmt(ib[1], "%.1f") if ib else "-")))
    w("")

    def both(label, key, spec, pick=None, only=None):
        fa = a.extreme(key, pick) if pick else a.median_of(key, only)
        fb = b.extreme(key, pick) if pick else b.median_of(key, only)
        w("%-26s %-18s %-18s" % (label, fmt(fa, spec), fmt(fb, spec)))

    w("%-26s %-18s %-18s" % ("outcome quality", a.name, b.name))
    both("final error (median m)", "final_error", "%.3f")
    both("min clearance (worst m)", "min_clearance", "%.3f", pick=min)
    both("path length (median m)", "path_length", "%.2f")
    # Speed is shown twice on purpose.  Achieved is the robot; commanded is the
    # planner.  When they diverge the planner is asking for motion it is not
    # getting, which is a different failure from going slowly -- and averaging
    # only one of them hides exactly that case.
    def speed_row(label, getter):
        vals = []
        for run in (a, b):
            v = [getter(t) for t in run.trials]
            v = [x for x in v if x is not None]
            vals.append(med(v) if v else None)
        w("%-26s %-18s %-18s" % (label, fmt(vals[0], "%.3f"),
                                 fmt(vals[1], "%.3f")))

    speed_row("mean speed achieved (m/s)", a.achieved_speed)
    speed_row("mean speed commanded (m/s)", a.commanded_speed)
    both("near-miss samples (med)", "n_near_miss", "%.0f")
    w("")

    # ------------------------------------------------------------ by target
    # One row per point in the world config, not per trial.  The cell is k/n and
    # the median arrival time of the trials that made it -- reported with its n
    # because with a handful of draws per target the difference between 1/3 and
    # 2/3 is not a difference.
    w("per target (indices are the order of `goals:` in the world config)")
    w("%-4s %-20s %-21s %-21s" % ("i", "target", a.name, b.name))
    w("-" * 68)
    solved = {"both": 0, "neither": 0, "a_only": 0, "b_only": 0}
    for gi in sorted(set(a.by_goal) & set(b.by_goal)):
        g = a.target(gi)
        tgt = "(%.2f, %.2f) %.1fm" % (g["x"], g["y"], g.get("distance") or 0.0)

        def cell(run):
            k, n = run.success_rate(gi)
            ts = run.by_goal[gi]
            durs = [t["duration"] for t in ts if t["verdict"] == "success"]
            extra = ("%.1fs med" % med(durs)) if durs else "-"
            return "%d/%d  %s" % (k, n, extra)

        ka = a.success_rate(gi)[0]
        kb = b.success_rate(gi)[0]
        if ka and kb:
            solved["both"] += 1
        elif ka:
            solved["a_only"] += 1
        elif kb:
            solved["b_only"] += 1
        else:
            solved["neither"] += 1
        w("%-4d %-20s %-21s %-21s" % (gi, tgt, cell(a), cell(b)))
    w("")
    w("%-26s %-18s %-18s" % ("trials per target", a.name, b.name))
    w("%-26s %-18s %-18s" % ("  in this order",
                             str(balance_note(a)), str(balance_note(b))))
    w("")
    w("same target, unpaired -- targets each planner was scored on, by whether")
    w("it ever solved them.  Trials are not matched: the moving obstacles are")
    w("re-randomized per trial, so nothing below is a paired result.")
    w("%-26s %d" % ("solved by both", solved["both"]))
    w("%-26s %d" % ("solved by neither", solved["neither"]))
    w("%-26s %d" % ("only %s" % a.name, solved["a_only"]))
    w("%-26s %d" % ("only %s" % b.name, solved["b_only"]))
    w("")

    # --------------------------------------------------------- the caveats
    w("known asymmetries -- not normalised, and they are not symmetric")
    w("-" * 68)
    for line in [
        "dynamic obstacles: the two runs use independent RVO trajectories;",
        "  matching goals do not imply matching obstacle encounters.",
        "collision: model-pose bounding circles are a proximity proxy,",
        "  especially for person meshes; they are not Gazebo contact events.",
        "footprint: both planners are scored with the same ground-truth robot,",
        "  but cspace's POLICY was trained to dilate obstacles by a 1.20 x 0.14 m",
        "  footprint while the Limo is 0.29 x 0.22 m.  Changing it would take the",
        "  checkpoint off its training distribution, so it stands: cspace keeps",
        "  ~0.6 m of fore/aft clearance and is genuinely handicapped in tight",
        "  gaps.  This is the one asymmetry config alone cannot remove.",
        "lidar FOV: the Limo scans 240 deg; cspace trained on 360 deg and has",
        "  allow_reverse enabled, so its rear 120 deg is blind here.",
        "control rate: cspace 10 Hz, neupan 50 Hz.  Reported, not equalised.",
        "speed caps: cspace +/-0.5 m/s; neupan ref_speed 0.3.",
        "stop thresholds: cspace stops at 0.3 m, neupan at 0.2 m.  Both are",
        "  scored against the same success_radius, with final_error shown.",
    ]:
        w(line)

    text = "\n".join(out)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
