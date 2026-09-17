# planner_bench

One arena, one robot, one goal set — so that **cspace** (`sac_cspace_ros`) and
**NeuPAN** (`neupan_ros`) can be compared without the comparison being decided by
whose simulation it is.

Before this package the two planners lived in incompatible worlds: cspace wanted
`multi_robot_scenario`'s Pioneer 3-DX, NeuPAN wanted the Limo in a world with ten
dynamic clones, the obstacles differed, and each run picked its own goals. Any
difference between them was uninterpretable.

```
roslaunch planner_bench bench.launch planner:=cspace
roslaunch planner_bench bench.launch planner:=neupan
```

## The fairness contract

The only thing that differs between the two runs is the token after `planner:=`.

| held equal | how |
|---|---|
| world | `worlds/bench_arena.world`, identical file, identified by sha1 |
| robot | the Limo, same URDF, same spawn pose, plugins driven through `/cmd_vel` + `/odom` |
| goals | one committed file, `config/goal_sets/bench_seed0.json`, read by both |
| scoring | ground-truth pose from `/gazebo/model_states`, never the planner's own odometry |
| clock | starts at **motion onset** (displacement > 0.05 m), not at goal publication |

`report.py` refuses to print a comparison whose runs disagree on world, goal set,
start pose, trial count or goal indices. A success rate measured against a
different set of goals is not a comparison, it is two unrelated measurements
printed next to each other.

The world hash each run records is **computed by the harness from the file it was
actually scored against**, not copied from the goal set. Those are two different
claims and they can disagree: the goal set is a static file recording what the
world hashed to when `gen_goals.py` last ran, so regenerating it after a run has
started leaves that run reporting a hash for a world it never loaded — which
makes `report.py` refuse two runs that in fact used byte-identical worlds, or
worse, accept two that did not. `bench.py` logs a warning and stores both values
(`world_sha1` and `goal_set_world_sha1`) when they disagree.

Timing is reported as **median and IQR, never mean**, with the denominator shown.
Time-to-goal is right-censored — a planner that never arrives has no arrival time,
and averaging only the arrivals flatters whichever planner failed more.

## Read this before interpreting the numbers

These are not normalised and they are not symmetric. They are printed at the
bottom of every `report.py` table for that reason.

1. **Footprint mismatch — the one thing config cannot fix.** Both planners are
   *scored* against the same ground-truth Limo (0.29 × 0.22 m), but cspace's
   *policy* was trained to dilate obstacles by a 1.20 × 0.14 m footprint. Changing
   it would take the checkpoint off its training distribution, so it stands: cspace
   holds ~0.6 m of fore/aft clearance and is genuinely handicapped in tight gaps.
   Mitigated (not removed) by keeping every corridor ≥ 1.6 m and every goal ≥ 0.8 m
   clear of obstacles. Fixing it properly means retraining cspace at Limo size.
2. **Lidar FOV.** The Limo scans 240°; cspace trained on 360° and has
   `allow_reverse` enabled, so its rear 120° is blind here. The FOV is hardcoded in
   `limo_gazebo_sim`'s `gazebo_laser` macro, so widening it means vendoring a second
   Limo URDF.
3. **Control rate.** cspace 10 Hz, NeuPAN 50 Hz. Reported, not equalised.
4. **Speed caps.** cspace ±0.5 m/s; NeuPAN `ref_speed` 0.3, `max_speed` [1, 3.14].
5. **Stop thresholds.** cspace stops at 0.3 m, NeuPAN at `arrive_threshold` 0.2 m.
   Both are scored against the same `success_radius`, with `final_error` reported
   alongside so the threshold cannot hide a difference.
6. **`pan:` block is inert.** `neupan_core.py` passes `kwargs={'pan': {...}}` and
   `neupan.py` does `config.update(kwargs)`, which *replaces* the whole `pan` dict
   rather than merging. So `pan.iter_num` / `dune_max_num` / `nrmp_max_num` /
   `iter_threshold` in `config/neupan_limo_bench.yaml` have no runtime effect. The
   shipped values happen to equal the library defaults, which is why this is not
   visible in the results.

## The physics landmine

**`worlds/bench_arena.world` must keep `max_step_size 0.01` /
`real_time_update_rate 100`.** This is not cosmetic and it cost a long time to
find.

At 1 ms the Limo's ODE contacts blow up on impact: `/gazebo/link_states` shows
`limo::base_footprint` going from rest to z = 20.95 m at |v| = 219180 m/s within
20 ms, after which every pose reads NaN and Gazebo reports the model at an exact
identity pose permanently. `set_model_state` then returns **success** and does
nothing.

That signature masquerades as a planner bug: the planner commands a real velocity,
the robot does not move, so every trial is scored `no_motion` and it looks like a
planner that never publishes `/cmd_vel`. Six of six full-speed wall impacts
exploded at 1 ms; zero of six at 10 ms. 7 of the 8 worlds shipped in
`limo_gazebo_sim` use 0.01/100.

`bench.py` logs the `SetModelState` response and the ground-truth pose before and
after every teleport for exactly this reason. A teleport that reports success while
the pose does not change means the model has gone inert — a different failure from
a rejected request, and one that must not be allowed to look like one.

## Layout

```
worlds/bench_arena.world    10x10 m, 11 static AABBs: 4 boundary walls, a central
                            divider with one 2 m doorway, a dead-end pocket, 3 boxes.
                            Pure primitives, no plugins, no model:// URIs.
launch/sim.launch           Gazebo + world + Limo + controllers + robot_state_publisher
launch/cspace.launch        the cspace node alone, pointed at config/sac_cspace_limo.yaml
launch/neupan.launch        the NeuPAN node alone, no rviz
launch/bench.launch         entry point: planner:=cspace|neupan, optionally + harness
config/bench.yaml           thresholds and the fairness contract, authored by hand
config/goal_sets/*.json     committed goal sets; both planners read the same file
config/sac_cspace_limo.yaml cspace IO/frames for the Limo (geometry kept verbatim)
config/neupan_limo_bench.yaml  NeuPAN config, `waypoints: []` and `loop: False`
src/bench_common.py         world parsing, ground-truth collision, URDF footprint
scripts/gen_goals.py        seeded goal generation + the negative control
scripts/bench.py            the harness: handshake, teleport, measure, verdict
scripts/report.py           side-by-side comparison; pure stdlib, runs without ROS
scripts/run_bench.sh        driver: reset, run a planner, wait, print verdicts
scripts/reset_ros.sh        kill ROS/Gazebo and prove it, before each run
reports/*_run.json          the harness output that report.py reads
```

## Running it

```bash
cd catkin_ws && catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5
source /opt/ros/noetic/setup.bash && source devel/setup.bash

# everything, in order: reset, cspace, reset, neupan, comparison table
rosrun planner_bench run_bench.sh both
```

Or step by step, if you want to watch each stage:

```bash
rosrun planner_bench gen_goals.py --check      # goal set reproduces (committed)
roslaunch planner_bench bench.launch planner:=cspace    # add gui:=false headless
roslaunch planner_bench bench.launch planner:=neupan
rosrun planner_bench report.py
```

**Reset between runs.** `run_bench.sh` does it for you; doing it by hand means
`rosrun planner_bench reset_ros.sh`, and it must print `clean` before you start.
A rosmaster or gzserver left over from the previous planner does not announce
itself as a port conflict — it attaches the new run to the old master and you get
a robot that does not move and a report full of `no_motion` that looks exactly
like a planner bug.

Useful overrides, all passed straight through to `roslaunch`:

```bash
# the negative control, into a scratch directory so it cannot overwrite the
# benchmark's own reports
PLANNER_BENCH_REPORT_DIR=/tmp/negctl rosrun planner_bench run_bench.sh both \
    goal_set:=$(rospack find planner_bench)/config/goal_sets/negcontrol.json

roslaunch planner_bench bench.launch planner:=cspace gui:=true   # watch it
roslaunch planner_bench bench.launch planner:=neupan harness:=false
```

`harness:=false` brings up the sim and the planner without the harness, which is
what you want when driving trials by hand or attaching `bench.py` to an
already-running sim. `gui:=false` is the default in `run_bench.sh` and worth
keeping when timing anything — the GUI steals CPU from the physics and sim time
is what the trials are measured in.

One trial is up to `trial_timeout` (90 s) of sim time, which headless runs at
roughly wall-clock rate, so budget ~90 s per goal plus ~20 s of startup: a full
11-goal run is about 20 minutes, the three-goal control about 5.

## Verifying the bench itself

A benchmark that can only report success is not measuring anything. Each of these
fails loudly if the machinery underneath is wrong.

```bash
python3 src/bench_common.py --selftest      # world parses, AABBs exact, bounds match
                                            # bench.yaml, corridors >= 1.6 m
python3 scripts/gen_goals.py --check        # goal set reproduces byte-for-byte
python3 scripts/gen_goals.py --selfcheck    # negative control goal set
python3 scripts/report.py --help
```

The `--selfcheck` goal set is the important one. It holds three goals, ordered by
how airtight the expectation is:

| goal | where | expected | what the wrong verdict would prove |
|---|---|---|---|
| 0 | borrowed from `bench_seed0.json`, los 1.40 m | `success` | a `stuck` here means the harness fails everything and measures nothing |
| 1 | (3.6, −3.6), behind the divider, no line of sight | `timeout`/`stuck` | ambiguous on its own — the planner could have found the 2 m doorway |
| 2 | (7.0, 0.0), 1.75 m **outside the east wall** | `timeout`/`stuck` | the criterion is vacuous |

Both planners pass it:

| planner | goal 0 (reachable) | goal 1 (walled off) | goal 2 (outside the arena) |
|---|---|---|---|
| cspace | `success`, 0.281 m | `stuck`, 4.69 m | `stuck`, 7.94 m |
| neupan | `success`, 0.161 m | `stuck`, 3.73 m | `stuck`, 5.18 m |

The goals that must not succeed are the ones that carry the argument, and neither
planner succeeds on either of them.

Goal 2 is the load-bearing one, and it is the reason goal 1 alone is not enough.
"Behind the divider" is a *soft* control: a `success` there would be readable two
ways, because the arena does contain a doorway and a planner that happened to find
it would be scoring a real success. The harness cannot tell that apart from a
vacuous criterion. A goal outside the boundary wall cannot be excused that way —
the robot cannot leave the arena, so the closest it can ever come is 2.0 m against
a 0.5 m `success_radius`. Success there is proof, not ambiguity.

Goal 0 is the opposite guard: without it, a harness that scored *everything*
`timeout` would look like a perfect negative control.

**Goal 0 is copied out of `bench_seed0.json`, not hand-placed.** An earlier version
wrote it by hand at 4.6 m — past the 3.6 m that `goal.max_distance` exists to
enforce — and cspace stalled on it. The control then reported "the harness cannot
pass anything", which was true of the goal and false of the harness. A goal that
both planners already reached under the benchmark's own rules is reachable by
demonstration; this one is the most open goal in the set, which makes the guard as
strong as it can be.

`bench_common.py --selftest` cross-checks its vendored `parse_world_obstacles` and
`check_pos` against the upstream `SAC/laser_env.py`. It needs `--sac-dir` to point
at a real SAC checkout; without one it skips the cross-check rather than passing
silently. Run it inside the container, where `/DRL-robot-navigation/SAC` exists.

## Speed is reported twice, and the gap is the finding

`mean speed achieved` is differenced from the ground-truth trace; `mean speed
commanded` averages `/cmd_vel`. These are two different quantities and an earlier
version of this harness conflated them under one name.

The conflation was not cosmetic. cspace, handed a goal outside the 2.4–3.6 m band
it trained on, holds a command of `v = -0.099, w = 1.144` indefinitely while the
robot does not move: 0.03 m of position span across 50 s of a 90 s trial, wheels
commanded, wheels not turning. Averaged as commanded velocity that reads as a
healthy 0.18 m/s, which is exactly the wrong story — the planner is not going
slowly, it is asking for motion it is not getting. The same trial differenced from
ground truth reads 0.00.

This is why `mean_speed` in the JSON is the achieved value and the commanded value
is kept beside it as `mean_cmd_speed`. `report.py` prints both rows, and recovers
the achieved speed from the trace for reports written before the split rather than
trusting the mislabelled field.

## Notes and known rough edges

- **`TF_REPEATED_DATA` warnings are expected.** The skid-steer plugin publishes
  `odom -> base_footprint` at the physics rate, so some timestamps collide and tf2
  drops the newer message, keeping the one already in the buffer. Measured over a
  full 11-goal run: 2717 warnings across 203 s of sim time, ~13/s, so roughly 13%
  of the plugin's tf samples are discarded and tf effectively runs near 87 Hz
  instead of 100. Harmless for planners reading at 10 Hz (cspace) and 50 Hz
  (NeuPAN), and symmetric — both read the same tf. The cost is log volume. Fixing
  it properly means giving the plugin an update rate that does not divide evenly
  into the physics step.
- **cspace stalls, permanently, on goals outside its training distance band.** Not
  a slow approach — a hard stop with the wheels commanded. It is the reason
  `goal.max_distance` is 3.6 m and why the generator rejects longer goals rather
  than merely discouraging them. A goal at 4.6 m produced 58–75 s of commanded
  motion with no displacement, and it did not recover within the trial. Teleports
  still worked and the wheel controllers were loaded identically, so this is the
  policy, not the sim. Nothing in this harness can fix it; goals outside the band
  are out of distribution for cspace and the goal set is built so none of them
  occur. It is also why the negative control's reachable goal is borrowed from the
  committed set rather than hand-placed.
- **Regenerate goals only through `gen_goals.py`.** It is the only thing that
  enforces the rejection rules (inside an obstacle, no line of sight, out of the
  distance band, too close to the start or to another goal). A hand-edited goal
  file silently voids the comparison.
- **The dead-end pocket is a hazard, never a destination.** No goal is ever placed
  behind it: both planners are structurally local (cspace is reactive on the raw
  goal bearing, NeuPAN builds a *straight* initial path), so such a goal is
  unsolvable by construction and would measure an architectural limit rather than
  navigation quality. It stays in the world as something a bad policy can fall
  into.
- **`map_frame` is `odom` for both planners** and there is no `map` frame in
  `sim.launch`. If anyone flips NeuPAN's to `map`, its tf lookup fails and it
  publishes no velocity at all — it does not fall back, it just spins at 50 Hz
  doing nothing.
- **Not in scope here:** fixing `rvo_node`'s SIGSEGV, dynamic obstacles, widening
  the Limo lidar to 360°, retraining cspace at Limo size, and the dead `+ 4`
  constant at `SAC/Sparrow_V2.py:1253`.

## This package is additive

It modifies no upstream clone (`rvo_ros`, `limo_ros`, `neupan_ros`,
`sac_cspace_ros`) and does not touch `multi_robot_scenario`. cspace's `/r1` topic
prefix appears only in its config YAML, so passing this package's own
`sac_cspace_limo.yaml` is enough; NeuPAN's launch file mirrors the upstream node
block but points `config_file` at this package's copy.
