#!/usr/bin/env python

"""Validate checkpoints against a config profile, without ROS topics or a master.

For each model it reports the sniffed architecture, whether the state dict loads,
and the shape of one observation -> action step, so a config/checkpoint mismatch
is caught before a rollout instead of mid-navigation.

  rosrun sac_cspace_ros check_models.py                       # all cspace models vs the sparrow config
  rosrun sac_cspace_ros check_models.py --config .../sac_cspace_legacy_sparrow.yaml
  rosrun sac_cspace_ros check_models.py --models SAC_sparrow_cspace_2 dense_sample_cspace_0
"""

import argparse
import glob
import math
import os
import sys

# installed, the script sits next to sac_cspace_core.py in lib/<pkg>/; run from
# the source tree, the module is one directory over in src/
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, os.pardir, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import yaml

from sac_cspace_core import (KINEMATIC_DIM, _ActorDeterministic, _ActorStochastic,
                             _arch_of, _BackboneNew, _BackboneOld, _kin_dim_of)

# models this node supports are the CNN c-space actors; the MLP SAC_laser_* and
# the *_critic.pth files are not
SKIP_PREFIX = ("SAC_laser", "Global_")


def discover(model_dir):
    names = []
    for path in sorted(glob.glob(os.path.join(model_dir, "*_actor.pth"))):
        name = os.path.basename(path)[:-len("_actor.pth")]
        if not name.startswith(SKIP_PREFIX):
            names.append(name)
    return names


def check(name, cfg, sac_dir):
    """Build the network for `name` under geometry `cfg`.  Returns (ok, detail)."""
    state = torch.load(os.path.join(cfg["model_dir"], "%s_actor.pth" % name),
                       map_location="cpu")

    try:
        backbone, head, feature_dim = _arch_of(state)
    except ValueError as exc:
        return None, str(exc)

    backbone_cls = _BackboneNew if backbone == "new" else _BackboneOld
    actor_cls = _ActorStochastic if head == "stochastic" else _ActorDeterministic
    actor = actor_cls(
        backbone_cls(in_channels=1, h_beams=cfg["cost_ray_num_beams"],
                     v_layers=cfg["nz"], feature_dim=feature_dim),
        feature_dim, KINEMATIC_DIM, 2)
    detail = "backbone=%s head=%s feature_dim=%d" % (backbone, head, feature_dim)

    # A checkpoint from before the previous-action feedback was removed is not a
    # config mismatch but a stale model: report it as a FAIL (not a skip) so
    # "3/11 ok" cannot quietly mean "3 loadable".
    try:
        _kin_dim_of(state, feature_dim, name)
    except ValueError as exc:
        return False, "%s -- %s" % (detail, exc)

    want = state["cnn.fc.weight"].shape[1]
    got = actor.cnn.fc.weight.shape[1]
    if want != got:
        return False, ("%s -- geometry mismatch: checkpoint wants fc flat %d, "
                       "nz=%d x %d beams gives %d"
                       % (detail, want, cfg["nz"], cfg["cost_ray_num_beams"], got))

    actor.load_state_dict(state)
    actor.eval()

    # one real observation -> action, through the same helper pipeline the node uses
    sys.path.insert(0, sac_dir)
    from cspace_cost_rays import (build_cspace, cost_rays_3d, make_footprint_masks,
                                  preprocess_cost_rays)

    g_min, nz, beams = cfg["g_min"], cfg["nz"], cfg["cost_ray_num_beams"]
    masks = torch.as_tensor(
        make_footprint_masks(g_min, cfg["a"], nz, cfg["footprint"]), dtype=torch.float32)
    R = int(math.ceil(cfg["cost_ray_max_range"] / g_min)) + 2
    G = 2 * R + 1
    occ = torch.zeros((1, G, G), dtype=torch.bool)
    occ[0, R + 10, R + 3] = True                    # a single obstacle 1 m ahead
    cspace = build_cspace(occ, masks)
    rays = cost_rays_3d(cspace, torch.full((1, 2), R, dtype=torch.long),
                        num_beams=beams, h_res_deg=cfg["cost_ray_h_res_degrees"],
                        max_range=cfg["cost_ray_max_range"], g_min=g_min)
    cr = preprocess_cost_rays(rays, max_range=cfg["cost_ray_max_range"]).reshape(1, -1)

    # kinematic block: [distance (m), heading (rad)], same as the node's _build_state
    kin = torch.zeros(1, KINEMATIC_DIM)
    kin[0, 0], kin[0, 1] = 2.0, 0.1
    with torch.no_grad():
        action = actor.get_mean_action(cr.reshape(1, 1, beams, nz), kin)
    a = action.reshape(-1)
    if a.numel() != 2 or not torch.isfinite(a).all() or (a.abs() > 1.0 + 1e-6).any():
        return False, "%s -- bad action %s" % (detail, a.tolist())

    return True, ("%s | rays %s -> state %d (+%d kinematic) | action [%.3f, %.3f]"
                  % (detail, tuple(rays.shape), cr.shape[1], KINEMATIC_DIM,
                     a[0], a[1]))


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    default_cfg = os.path.join(here, "..", "config", "sac_cspace_sparrow.yaml")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_cfg)
    parser.add_argument("--sac-dir", default="/DRL-robot-navigation/SAC")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--models", nargs="*", default=None,
                        help="default: every *_actor.pth in the model dir")
    args = parser.parse_args(argv)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg["model_dir"] = args.model_dir or os.path.join(args.sac_dir, "pytorch_models")

    names = args.models or discover(cfg["model_dir"])
    print("config: %s" % args.config)
    print("  nz=%d a=%s g_min=%s range=%s beams=%s ray_frame=%s preprocess=%s"
          % (cfg["nz"], cfg["a"], cfg["g_min"], cfg["cost_ray_max_range"],
             cfg["cost_ray_num_beams"], cfg.get("ray_frame"), cfg.get("preprocess")))
    print()

    failed = 0
    for name in names:
        try:
            ok, detail = check(name, cfg, args.sac_dir)
        except Exception as exc:                    # noqa: BLE001 - report, don't crash
            ok, detail = False, "%s: %s" % (type(exc).__name__, exc)
        mark = {True: "PASS", False: "FAIL", None: "skip"}[ok]
        if ok is False:
            failed += 1
        print("[%s] %-28s %s" % (mark, name, detail))

    print()
    print("%d/%d ok" % (len(names) - failed, len(names)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
