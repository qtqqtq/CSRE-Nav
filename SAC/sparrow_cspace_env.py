"""
Sparrow-V2 environment that outputs the DRL-robot-navigation CNN (c-space) state.

This "stitches" the NavRL-style configuration-space cost-ray pipeline
(`cspace_cost_rays.py`) into ColorDynamic's GPU-vectorized `Sparrow` simulator,
replacing the Gazebo/c_space_mapper ROS pipeline.  The observation is exactly the
364-dim CNN state consumed by `cnn_network.py` / `train_cnn_sac.py`:

    state = flatten( preprocess(cost_rays) ) + [distance, heading, v_lin, v_ang]
              |___________ 360 ___________|    |___________ 4 ____________|

matching `laser_env.step()` (CNN mode), where:
    cost_rays : (36 beams x 10 theta layers) NavRL cost rays, metres
    distance  : distance to goal, metres
    heading   : heading error (robot heading -> goal), radians
    v_lin     : linear velocity command  (Continuous: a[:,0] in [-1,1])
    v_ang     : angular velocity command (Continuous: a[:,1] in [-1,1])

Coordinate / unit mapping (Sparrow is pygame y-down, cm; the c-space is metre):
    - Sparrow world -> robot frame:
          rel_x = (dx*cos_t - dy*sin_t) / 100
          rel_y = (dx*sin_t + dy*cos_t) / 100
      where (dx,dy) = point - robot (cm), t = robot heading, forward = (cos t, -sin t).
      The robot frame +x is the robot's forward, +y its left (verified).
    - The robot footprint is a regular 16-gon approximating Sparrow's collision
      circle (`collision_trsd = car_radius + 5` cm).  Override via
      `cspace_cfg={'footprint': [[x,y], ...]}` (metres).

The obstacle occupancy grid is rasterised from Sparrow's obstacle *boundary*
points (`vec_static_obs_P_shaped`, `vec_dynamic_obs_P_shaped`,
`vec_map_obs_P_shaped`) around the robot.  Boundary-only is sufficient: a cost ray
always stops at the first (near) obstacle surface, and footprint-overlap with a
convex obstacle always touches its boundary.

Run the self-test inside the container:
    python SAC/sparrow_cspace_env.py
"""
from __future__ import annotations

import math
import os
import sys

import torch

# ---------------------------------------------------------------------------
# Imports: make both the ColorDynamic dir and this (SAC) dir importable.
# ---------------------------------------------------------------------------
_SAC_DIR = os.path.dirname(os.path.abspath(__file__))          # .../SAC
_COLORDYNAMIC_DIR = os.path.join(_SAC_DIR, "..", "ColorDynamic")
for _p in (_COLORDYNAMIC_DIR, _SAC_DIR):
    _p = os.path.normpath(_p)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from Sparrow_V2 import Sparrow, Sparrow_PlayGround                # noqa: E402
from cspace_cost_rays import (                                    # noqa: E402
    make_footprint_masks,
    build_cspace,
    cost_rays_3d,
    preprocess_cost_rays,
)
import settings                                                   # noqa: E402

__all__ = ["SparrowCspaceEnv", "SparrowCspacePlayground", "SPARROW_CSPACE_CFG"]

# c-space defaults for Sparrow (metres; resolution/range adapted to the 9 cm
# robot vs. the 0.44 m Pioneer).  nz=10 & 36 beams keep the CNN shape identical
# to the Gazebo pipeline (RAY_V_LAYERS / RAY_H_BEAMS in settings.py).
SPARROW_CSPACE_CFG = dict(
    g_min=0.1,                    # c-space cell size, m (10 cm)
    a=0.698,                      # theta resolution, rad  -> nz = 10
    nz=10,                        # number of theta layers
    cost_ray_max_range=4.0,       # m  (= Sparrow's local horizon D = 400 cm)
    cost_ray_num_beams=36,
    cost_ray_h_res_degrees=10.0,
)


class SparrowCspaceEnv(Sparrow):
    """Sparrow-V2 with a c-space (NavRL) observation.

    Use exactly like `Sparrow`, but `reset()`/`step()` return the 364-dim
    c-space CNN state instead of the default lidar state.  Defaults to the
    Continuous action (action_dim=2); set `action_type='Discrete'` for the
    7-way discrete action (the last two state terms then hold the mapped
    `a_state` velocities).
    """

    def __init__(self, **params):
        cspace_cfg = params.pop("cspace_cfg", None)
        self.cspace_cfg = dict(SPARROW_CSPACE_CFG)
        if cspace_cfg:
            self.cspace_cfg.update(cspace_cfg)

        params.setdefault("action_type", "Continuous")
        params.setdefault("render_mode", None)
        params.setdefault("dynamic_obs", False)   # c-space 环境默认关闭动态障碍物
        super().__init__(**params)

        # footprint: from settings (circular collision footprint) unless overridden
        self.cspace_cfg.setdefault("footprint", settings.ROBOT_FOOTPRINT)

        # one-time footprint masks + grid geometry
        self._cspace_masks = torch.as_tensor(
            make_footprint_masks(
                self.cspace_cfg["g_min"],
                self.cspace_cfg["a"],
                self.cspace_cfg["nz"],
                self.cspace_cfg["footprint"],
            ),
            dtype=torch.float32,
            device=self.dvc,
        )
        self._cR = int(math.ceil(
            self.cspace_cfg["cost_ray_max_range"] / self.cspace_cfg["g_min"]
        )) + 2
        self._cG = 2 * self._cR + 1
        self._robot_cell = torch.full(
            (self.N, 2), self._cR, dtype=torch.long, device=self.dvc
        )

        # output state dims
        self.cost_ray_num_beams = self.cspace_cfg["cost_ray_num_beams"]
        self.cost_ray_nz = self.cspace_cfg["nz"]
        self.cspace_state_dim = self.cost_ray_num_beams * self.cost_ray_nz + 4

    # ------------------------------------------------------------------ utils
    def _heading_error(self):
        """Heading error (robot heading -> goal) in radians, wrapped to [-pi, pi].

        Reuses Sparrow's pygame-world `beta` (angle of the goal direction) but
        keeps it un-normalised and wraps the result to [-pi, pi].
        """
        xy = self.car_state[:, 0:2] - self.target_point          # robot - goal
        beta = torch.atan2(xy[:, 0], xy[:, 1]) + torch.pi / 2    # goal dir angle
        alpha = beta - self.car_state[:, 2]                      # heading error
        return (alpha + torch.pi) % (2 * torch.pi) - torch.pi

    def _action_to_v(self, current_a):
        """Return the (N,2) velocity command held in the last two state terms."""
        if current_a is None:
            return torch.zeros(self.N, 2, device=self.dvc)
        if self.action_type == "Continuous":
            return current_a
        return self.a_state[current_a]                            # (N,2)

    # ------------------------------------------------------------------- core
    def _build_occ_grid(self):
        """Rasterise obstacle boundary points into a robot-centric grid (N,G,G).

        World (cm) -> robot frame (m) -> cell index at `g_min` resolution,
        robot always at the grid centre.  Out-of-window points are dropped.
        """
        pts = [self.vec_static_obs_P_shaped, self.vec_dynamic_obs_P_shaped]
        if getattr(self, "vec_map_obs_P_shaped", None) is not None:
            pts.insert(0, self.vec_map_obs_P_shaped)
        all_pts = torch.cat([p[:, :, :, 0].float() for p in pts], dim=1)  # (N,M,2) cm

        # world -> robot frame, metres
        dx = all_pts[:, :, 0] - self.car_state[:, 0, None]
        dy = all_pts[:, :, 1] - self.car_state[:, 1, None]
        cos_t = torch.cos(self.car_state[:, 2])[:, None]
        sin_t = torch.sin(self.car_state[:, 2])[:, None]
        rel_x = (dx * cos_t - dy * sin_t) / 100.0
        rel_y = (dx * sin_t + dy * cos_t) / 100.0

        g_min = self.cspace_cfg["g_min"]
        G = self._cG
        R = self._cR
        ci = (R + torch.round(rel_x / g_min)).long()
        cj = (R + torch.round(rel_y / g_min)).long()
        inb = (ci >= 0) & (ci < G) & (cj >= 0) & (cj < G)

        occ_flat = torch.zeros(self.N * G * G, dtype=torch.bool, device=self.dvc)
        n_idx = torch.arange(self.N, device=self.dvc)[:, None].expand_as(ci)[inb]
        occ_flat[n_idx * (G * G) + ci[inb] * G + cj[inb]] = True
        return occ_flat.view(self.N, G, G)

    def _get_cspace_state(self, current_a):
        """Build the 364-dim CNN state (N, 360 + 4)."""
        cfg = self.cspace_cfg
        occ = self._build_occ_grid()
        cspace = build_cspace(occ, self._cspace_masks)
        cost_rays = cost_rays_3d(
            cspace, self._robot_cell,
            num_beams=cfg["cost_ray_num_beams"],
            h_res_deg=cfg["cost_ray_h_res_degrees"],
            max_range=cfg["cost_ray_max_range"],
            g_min=cfg["g_min"],
            a=cfg["a"],
        )                                                          # (N,B,nz) m
        # stash raw / processed cost rays + c-space for logging & debugging
        self.cost_rays_raw = cost_rays.detach()                    # (N,B,nz) m
        self.occ_grid = occ.detach()                               # (N,G,G) bool
        self.cspace = cspace.detach()                              # (N,G,G,nz) bool
        cr_proc = preprocess_cost_rays(
            cost_rays, max_range=cfg["cost_ray_max_range"]
        )                                                          # (N,1,B,nz)
        self.cost_rays_proc = cr_proc.detach()
        cr = cr_proc.reshape(self.N, -1)                           # (N, B*nz)

        # fresh distance (post AutoReset) + heading + velocity command
        self.d2target_now = (
            self.car_state[:, 0:2] - self.target_point
        ).pow(2).sum(-1).pow(0.5)
        distance = (self.d2target_now / 100.0).unsqueeze(-1)       # (N,1) m
        heading = self._heading_error().unsqueeze(-1)              # (N,1) rad
        v = self._action_to_v(current_a)                           # (N,2)

        return torch.cat([cr, distance, heading, v], dim=1)        # (N, B*nz+4)

    def is_stuck(self, threshold=0.15):
        """Return (N,) bool: the robot is boxed in — every beam's nearest
        footprint hit is below `threshold` metres (cost rays ~0 in all
        directions).  Used to auto-reset episodes where the robot is wedged
        against obstacles and can only spin in place."""
        cr = getattr(self, "cost_rays_raw", None)
        if cr is None:
            return torch.zeros(self.N, dtype=torch.bool, device=self.dvc)
        per_beam = cr.min(dim=-1).values              # (N, B) nearest hit per beam
        return per_beam.max(dim=-1).values < threshold  # (N,) all beams jammed

    # ----------------------------------------------------------------- visualize
    def _plot_cspace(self, env_idx=0, save_path="cspace_viz.png"):
        """Render the occupancy grid, 3-D c-space and cost rays to a PNG.

        Headless-safe (matplotlib Agg).  `env_idx` selects which vectorized
        environment to draw; the grid is robot-centric (robot at the centre,
        +x = forward).  Returns the absolute path of the saved figure.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        cfg = self.cspace_cfg
        nz = cfg["nz"]
        R = self._cR

        occ = self.occ_grid[env_idx].detach().cpu().numpy()          # (G,G) bool
        cspace = self.cspace[env_idx].detach().cpu().numpy()         # (G,G,nz) bool
        cr_raw = self.cost_rays_raw[env_idx].detach().cpu().numpy()  # (B,nz) m

        ncol = 5
        nrow = math.ceil(nz / ncol)
        fig = plt.figure(figsize=(4 * ncol, 3 * nrow + 8))
        gs_top = GridSpec(1, 3, top=0.92, bottom=0.70, hspace=0.4, figure=fig)
        gs_bot = GridSpec(nrow, ncol, top=0.62, bottom=0.04,
                          hspace=0.35, wspace=0.25, figure=fig)

        # --- overview: occupancy grid / collapsed c-space / cost rays ---
        ax_occ = fig.add_subplot(gs_top[0, 0])
        ax_occ.imshow(occ, origin="lower", cmap="gray_r", interpolation="nearest")
        ax_occ.arrow(R, R, 4, 0, head_width=0.7, head_length=0.7,
                     fc="b", ec="b", length_includes_head=True)
        ax_occ.plot(R, R, "bo", ms=4, mfc="none")
        ax_occ.set_title("Occupancy grid\n(robot-centric, +x fwd)")
        ax_occ.set_xlabel("x cell")
        ax_occ.set_ylabel("y cell")

        ax_any = fig.add_subplot(gs_top[0, 1])
        ax_any.imshow(cspace.any(axis=-1), origin="lower", cmap="Reds",
                      vmin=0, vmax=1, interpolation="nearest")
        ax_any.arrow(R, R, 4, 0, head_width=0.7, head_length=0.7,
                     fc="b", ec="b", length_includes_head=True)
        ax_any.plot(R, R, "bo", ms=4, mfc="none")
        ax_any.set_title("C-space collapsed\n(any θ in collision)")
        ax_any.set_xlabel("x cell")
        ax_any.set_ylabel("y cell")

        ax_cr = fig.add_subplot(gs_top[0, 2])
        im = ax_cr.imshow(cr_raw.T, origin="lower", aspect="auto", cmap="hot",
                          interpolation="nearest")
        ax_cr.set_title("Cost rays (raw, m)")
        ax_cr.set_xlabel("beam")
        ax_cr.set_ylabel("theta layer")
        plt.colorbar(im, ax=ax_cr, shrink=0.8)

        # --- c-space slices, one per theta layer ---
        for k in range(nz):
            ax = fig.add_subplot(gs_bot[k // ncol, k % ncol])
            ax.imshow(cspace[:, :, k], origin="lower", cmap="Reds",
                      vmin=0, vmax=1, interpolation="nearest")
            ax.plot(R, R, "bo", ms=3, mfc="none")
            ax.set_title("θ = %.2f rad" % (k * cfg["a"]))
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(
            "SparrowCspaceEnv c-space (env %d, g_min=%.2f m, %d θ layers)"
            % (env_idx, cfg["g_min"], nz), fontsize=12,
        )
        fig.savefig(save_path, dpi=110)
        plt.close(fig)
        return os.path.abspath(save_path)

    # ------------------------------------------------------------- gym-like API
    def reset(self):
        _, info = super().reset()                     # base state discarded
        return self._get_cspace_state(current_a=None), info

    def step(self, current_a):
        _, reward, dw, tr, info = super().step(current_a)
        return self._get_cspace_state(current_a=current_a), reward, dw, tr, info


class SparrowCspacePlayground(Sparrow_PlayGround, SparrowCspaceEnv):
    """Interactive c-space playground.

    Combines `Sparrow_PlayGround`'s mouse-driven waypoint navigation (click the
    rendered map to add a target waypoint, press Esc to reset, human pygame
    rendering) with `SparrowCspaceEnv`'s 364-dim c-space (NavRL) observation, so
    it can be driven by any `CNNSAC` model in `pytorch_models/`.

    MRO (`Sparrow_PlayGround` first) reuses the playground's `_render_frame`,
    `_reward_function`, `_AutoReset` and `__init__` (waypoints / mouse), while
    `reset()`/`step()` below return the c-space state instead of the raw lidar
    state.  Must be instantiated with `N=1` (the playground renders one world).
    """

    def reset(self):
        _, info = Sparrow_PlayGround.reset(self)         # waypoint/mouse env reset
        return self._get_cspace_state(current_a=None), info

    def step(self, current_a):
        _, reward, dw, tr, info = Sparrow_PlayGround.step(self, current_a)
        return self._get_cspace_state(current_a=current_a), reward, dw, tr, info


# ---------------------------------------------------------------------------
# Self-test (run inside the container: `python SAC/sparrow_cspace_env.py`)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    # 使用各向异性、前后不对称的"车形"footprint(前窄后宽,箭头朝 +x 前进方向),
    # 相比默认近圆形足迹,能让各 θ 层的 c-space 切片差异更明显,便于观察旋转膨胀。
    env = SparrowCspaceEnv(
        N=2, O=3, dvc=torch.device("cpu"), render_mode=None,
        action_type="Continuous", max_ep_steps=50,
        cspace_cfg={"footprint": [
            [1.00,  0.06],   # 前右
            [0.20, -0.06],   # 前左
            [-0.16, -0.10],  # 后左(宽)
            [-0.20,  0.00],  # 后中(尾)
            [-0.16,  0.10],  # 后右(宽)
        ]},
    )
    s, info = env.reset()
    print("reset state shape:", tuple(s.shape), "(expect (2, 364))")
    print("cost-ray sub-range: [%.2f, %.2f] (expect [0, %.1f])" % (
        s[:, :360].min().item(), s[:, :360].max().item(),
        env.cspace_cfg["cost_ray_max_range"],
    ))
    print("distance col (m):", s[:, 360].tolist(), ">=0:", bool((s[:, 360] >= 0).all()))
    print("heading col (rad):", s[:, 361].tolist())

    for _ in range(3):
        a = torch.rand(env.N, 2, device=env.dvc) * 2 - 1       # [-1,1] x [-1,1]
        s_next, r, dw, tr, _ = env.step(a)
    print("step state shape:", tuple(s_next.shape), "dtype:", s_next.dtype)
    print("reward:", r.tolist(), "done:", dw.tolist())
    assert s.shape == (2, 364) and s_next.shape == (2, 364)
    assert s.dtype == torch.float32

    # visualize the generated c-space (occ grid + c-space slices + cost rays)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    png = env._plot_cspace(env_idx=0, save_path=os.path.join(out_dir, "cspace_viz.png"))
    print("c-space visualization saved to:", png)
    print("self-test OK")
