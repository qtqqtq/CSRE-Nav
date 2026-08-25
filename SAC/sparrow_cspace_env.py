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
    - The robot footprint is a thin rectangle (settings.ROBOT_FOOTPRINT, long
      axis along +x = forward) for the observation / cost rays; Sparrow's actual
      collision remains its `collision_trsd = car_radius + 5` cm circle.  Override
      via `cspace_cfg={'footprint': [[x,y], ...]}` (metres).

The obstacle occupancy grid is rasterised from the robot's *lidar* hit points
(`ld_scan_result` / `ld_vectors_wd`) — the robot's actual line-of-sight
perception, mirroring the Gazebo `lidar_to_occupancy_node` — rather than the
ground-truth obstacle boundary points (`vec_*_obs_P_shaped`).  Only cells hit by
a lidar ray within `ld_d_range` are marked occupied; everything else is free.

Run the self-test inside the container:
    python SAC/sparrow_cspace_env.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pygame
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
from global_planner import (                                      # noqa: E402
    GraphPlanner,
    build_world_occ_grid,
    draw_planned_path,
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
    cost_ray_max_range=3.0,       # m  (= Sparrow's local horizon D = 400 cm)
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

        # 全局规划引导（"Planning on the Graph"）：开启后每个 env 沿一条全局路径的
        # 航点推进，state 的 distance/heading 与 reward shaping 相对当前航点而非真目标。
        self.global_plan = bool(params.pop("global_plan", settings.GLOBAL_PLAN_ENABLED))

        params.setdefault("action_type", "Continuous")
        params.setdefault("render_mode", None)
        params.setdefault("dynamic_obs", settings.SPARROW_DYNAMIC_OBS)   # c-space 环境默认关闭动态障碍物
        params.setdefault("stuck_reset", settings.SPARROW_STUCK_RESET)    # 卡死(cost ray 全≈0)时自动重置
        params.setdefault("stuck_threshold", settings.SPARROW_STUCK_THRESHOLD)
        # 雷达探测距离必须覆盖 c-space 窗口：基于雷达的 occupancy grid 在
        # ld_d_range 之外会"失明"(Sparrow 默认只有 100 cm)。这里对齐 cost_ray 范围。
        params.setdefault(
            "ld_d_range",
            int(round(self.cspace_cfg["cost_ray_max_range"] * 100.0)),
        )
        super().__init__(**params)

        # 关闭 _render_frame 内部的 pygame.display.update()（手动 flush），由
        # play_cspace 主循环在所有叠加层绘制完成后统一 flush 一次，避免叠加层闪动。
        self.manual_flip = True

        # 全局规划引导状态：共享一张静态 c-space 图（N 个 env 障碍物相同），
        # 每 env 一条航点序列 plan_poses[i]（(x,y,θ) 列表）+ 当前航点下标 plan_idx[i]。
        self.wp_reach_cm = settings.PLANNER_WP_REACH_CM
        self.plan_poses = [None] * self.N          # None -> 无引导（直接追真目标）
        self.plan_idx = [0] * self.N
        if self.global_plan:
            # 规划器足迹默认 28x28cm（≈碰撞圆），与观测用的大足迹(settings.ROBOT_FOOTPRINT)
            # 不同——1.2m 足迹会挡住 > 一半地图。见 global_planner 模块 docstring。
            self.planner = GraphPlanner(
                g_min=settings.PLANNER_G_MIN,
                nz=settings.PLANNER_NZ,
                a=self.cspace_cfg["a"],
                footprint=settings.PLANNER_FOOTPRINT,
                stride_ij=settings.PLANNER_STRIDE_IJ,
                stride_k=settings.PLANNER_STRIDE_K,
                connect_dist_cm=settings.PLANNER_CONNECT_DIST_CM,
                min_wp_spacing_cm=settings.PLANNER_MIN_WP_SPACING_CM,
            )
        else:
            self.planner = None

        # footprint: from settings (rectangular footprint) unless overridden
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

    # --------------------------------------------- global-plan helpers
    def _rebuild_planner(self):
        """重建共享的静态 c-space 图（world_occ_full 在 reset() 中已更新）。

        障碍物只在完整 reset() 时重新随机，因此训练期间图只建一次；
        done/stuck 触发的逐 env 重规划只调 plan()（毫秒级），不重建图。
        """
        if not self.global_plan:
            return
        occ = build_world_occ_grid(self, self.planner.g_min)
        self.planner.build_graph(occ)

    def _plan_intermediate_mask(self):
        """(N,) bool：该 env 当前处于 plan 的中间航点（非 None 且非末点）。"""
        if not self.global_plan:
            return torch.zeros(self.N, dtype=torch.bool, device=self.dvc)
        return torch.tensor(
            [bool(self.plan_poses[i]) and self.plan_idx[i] < len(self.plan_poses[i]) - 1
             for i in range(self.N)],
            dtype=torch.bool, device=self.dvc,
        )

    def _advance_env_waypoint(self, i):
        """若机器人距当前航点 < wp_reach_cm 且非末点，推进到下一航点。

        wp_reach_cm(60) > target_area(30)，所以中间航点几乎不会被 win 命中，
        而是在到达前就切到下一航点；win 只可能在真目标(末点)处触发。
        """
        pp = self.plan_poses[i]
        if pp is None:
            return
        wp = self.wp_reach_cm
        while self.plan_idx[i] < len(pp) - 1:
            cur = self.target_point[i]
            d = (self.car_state[i, 0:2] - cur).pow(2).sum().pow(0.5).item()
            if d < wp:
                self.plan_idx[i] += 1
                nxt = pp[self.plan_idx[i]]
                self.target_point[i] = torch.tensor([nxt[0], nxt[1]], device=self.dvc)
            else:
                break

    def _advance_waypoints(self):
        """步首：对所有 env 按当前位姿推进航点（仅 global_plan 模式）。"""
        if self.global_plan:
            for i in range(self.N):
                self._advance_env_waypoint(i)

    def _replan(self, i):
        """对第 i 个 env 重新规划：从当前位姿到真目标，取航点序列。

        `plan()` 的起点/终点是临时顶点（用后即删），不会让图随重规划无界增长。
        无可行路径时回退无引导（target_point 保持真目标）。
        """
        if not self.global_plan:
            return
        car = self.car_state[i]
        goal = self.target_point[i]                       # 真目标（_target_point_init / _AutoReset 设定）
        res = self.planner.plan(
            (float(car[0]), float(car[1]), float(car[2])),
            (float(goal[0]), float(goal[1]), None),       # goal 朝向任意
        )
        if res is None or len(res.poses) < 2:
            self.plan_poses[i] = None
            self.plan_idx[i] = 0
            return
        self.plan_poses[i] = res.poses
        self.plan_idx[i] = 1
        nxt = res.poses[1]
        self.target_point[i] = torch.tensor([nxt[0], nxt[1]], device=self.dvc)
        self._advance_env_waypoint(i)                     # 跳过头几个过近的航点

    def _replan_after_reset(self, dw, tr, stuck):
        """对触发重置的 env（done / stuck）重新规划。

        这些 env 的位姿或真目标已被重置（_AutoReset / _reset_env），需要新的航点序列。
        """
        if not self.global_plan:
            return
        reset_mask = torch.zeros(self.N, dtype=torch.bool, device=self.dvc)
        for m in (dw, tr, stuck):
            if m is not None:
                reset_mask = reset_mask | m
        for i in reset_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
            self._replan(i)

    def _reward_function(self, current_a):
        """Base 奖励 + global-plan 中间航点的 win 门控。

        中间航点到达时不结束 episode、也不给 AWARD（航点推进由 _advance_waypoints
        在 wp_reach_cm>target_area 处提前完成，win 几乎只在真目标处触发；这里是
        防止航点过密等边角情形下的提前获胜）。
        """
        super()._reward_function(current_a)
        if not self.global_plan:
            return
        mid = self._plan_intermediate_mask()
        mid_win = self.win_vec & mid
        if mid_win.any():
            self.reward_vec = self.reward_vec.clone()
            self.reward_vec[mid_win] = 0.0
            self.win_vec = self.win_vec & ~mid_win
            self.dw_vec = self.dw_vec & ~mid_win
            self.done_vec = self.done_vec & ~mid_win

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
        """Rasterise lidar hit points into a robot-centric grid (N,G,G).

        Mirrors the Gazebo `lidar_to_occupancy_node` (LaserScan callback): only
        the robot's *line-of-sight* lidar hits are marked occupied — its actual
        perception — instead of the ground-truth obstacle boundary points
        (`vec_*_obs_P_shaped`).  A ray that reaches the lidar's max range
        (`ld_d_range`) hit nothing and is dropped; the grid defaults to free.

        World (cm) -> robot frame (m) -> cell index at `g_min` resolution,
        robot always at the grid centre.
        """
        hit = self.ld_scan_result < self.ld_d_range                 # (N, ld_num)
        # hit point world coords (cm): robot centre + distance * unit direction
        hit_pts = (
            self.car_state[:, None, 0:2]
            + self.ld_scan_result[:, :, None] * self.ld_vectors_wd
        )                                                            # (N, ld_num, 2)

        # world -> robot frame, metres
        dx = hit_pts[:, :, 0] - self.car_state[:, 0, None]
        dy = hit_pts[:, :, 1] - self.car_state[:, 1, None]
        cos_t = torch.cos(self.car_state[:, 2])[:, None]
        sin_t = torch.sin(self.car_state[:, 2])[:, None]
        rel_x = (dx * cos_t - dy * sin_t) / 100.0
        rel_y = (dx * sin_t + dy * cos_t) / 100.0

        g_min = self.cspace_cfg["g_min"]
        G = self._cG
        R = self._cR
        ci = (R + torch.round(rel_x / g_min)).long()
        cj = (R + torch.round(rel_y / g_min)).long()
        inb = hit & (ci >= 0) & (ci < G) & (cj >= 0) & (cj < G)

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

    def is_stuck(self, threshold=0.05):
        """Return (N,) bool: at theta=0 (driving straight, no rotation) even the
        most open beam's cost ray is below `threshold` metres — the robot is boxed
        in at its current heading and can only spin in place."""
        cr = getattr(self, "cost_rays_raw", None)
        if cr is None:
            return torch.zeros(self.N, dtype=torch.bool, device=self.dvc)
        cr0 = cr[:, :, 0]                              # (N, B) theta=0 (当前朝向, 不旋转)
        return cr0.max(dim=-1).values < threshold      # (N,) 最开阔方向仍 < threshold

    def _reset_env(self, i):
        """Re-init one (stuck) env — base-env semantics.

        The playground overrides this with its waypoint/mouse reset semantics.
        """
        self._car_loc_init(i)
        self._target_point_init(i)
        self.step_counter_vec[i] = 0

    def _apply_stuck_reset(self, dw):
        """Detect boxed-in envs (theta=0 cost rays ~0 in every direction) and
        reset them, OR-ing the flag into `dw` so the caller treats them as terminal.

        Returns `(dw, stuck)` where `stuck` is the (N,) bool mask (None when
        `stuck_reset` is disabled).  Callers must recompute the c-space state
        afterwards because `_reset_env` moves the robot.
        """
        if not self.stuck_reset:
            return dw, None
        stuck = self.is_stuck(self.stuck_threshold)
        if stuck.any():
            for i in stuck.nonzero(as_tuple=False).squeeze(-1).tolist():
                self._reset_env(i)
            # _reset_env 移动了小车，重新扫描雷达，否则基于雷达的 occupancy grid
            # 会使用过期(移动前)的 ld_scan_result / ld_vectors_wd。
            self._ld_scan_vec()
            dw = dw | stuck
        return dw, stuck

    def _AutoReset(self):
        """Reset done掉的 env（truncate / win / exceed）—— 碰撞不再触发重置。

        原 Sparrow._AutoReset 会在碰撞时 `_car_loc_init` 重置小车位姿；这里去除该
        分支：碰撞（卡死）只保留 `collide_vec`(= stuck-reset 掩码) 的 -200 惩罚信号，
        位姿重置交由 `_apply_stuck_reset`（卡死）处理。
        """
        if self.done_vec.any():
            DoneEnv_idx = torch.where(self.done_vec)[0]
            for i in DoneEnv_idx:
                self._target_point_init(i)
            self.step_counter_vec[self.done_vec] = 0

    # ----------------------------------------------------------------- visualize
    def _draw_robot(self):
        """Draw the c-space footprint polygon (vs. Sparrow's collision circle).

        Converts the footprint vertices from robot frame (metres, +x forward,
        +y left) to world cm and renders them as a polygon, so the on-screen
        robot body matches the footprint used by the cost-ray observation.
        Also draws the forward heading line.
        """
        car = self.car_state[self.ri]
        cx, cy, t = float(car[0]), float(car[1]), float(car[2])
        cos_t, sin_t = math.cos(t), math.sin(t)

        pts = []
        for fx, fy in self.cspace_cfg["footprint"]:
            # robot frame (m) -> world (cm); matches _build_occ_grid convention
            wx = cx + 100.0 * (fx * cos_t + fy * sin_t)
            wy = cy + 100.0 * (-fx * sin_t + fy * cos_t)
            pts.append(self._world_2_grid(torch.tensor([wx, wy])).numpy())
        pygame.draw.polygon(self.canvas, (200, 128, 250), pts)
        pygame.draw.polygon(self.canvas, (120, 40, 160), pts, width=2)

        # heading line (forward direction from centre)
        head = car[0:2].cpu() + self.car_radius * torch.tensor(
            [cos_t, -sin_t])
        pygame.draw.line(
            self.canvas,
            (0, 255, 255),
            self._world_2_grid(car[0:2]).cpu().numpy(),
            self._world_2_grid(head).numpy(),
            width=2,
        )

    def _draw_lidar_rays(self):
        """Replace the base lidar rays with c-space cost rays.

        For each of the 36 beam directions, draw a ray out to the *farthest*
        distance the robot can reach at any theta layer (`cost_rays_raw.max(dim=-1)`),
        i.e. the most open route in that direction.  Ray colour goes red (near /
        blocked) -> green (far / open).
        """
        raw = getattr(self, "cost_rays_raw", None)
        if raw is None:
            return
        max_d = raw[self.ri].max(dim=-1).values   # (B,) metres — farthest reach per beam
        car = self.car_state[self.ri]
        t = float(car[2])

        # robot-frame beam angles (rad) -> world (pygame y-down) direction
        beams = torch.arange(self.cost_ray_num_beams, device=self.dvc).float() \
            * math.radians(self.cspace_cfg["cost_ray_h_res_degrees"])
        world_ang = beams - t
        dir_wd = torch.stack((torch.cos(world_ang), torch.sin(world_ang)), dim=-1)  # (B,2)

        sta = car[0:2]
        ends = sta + (max_d * 100.0).unsqueeze(-1) * dir_wd          # (B,2) world cm
        sta_gd = self._world_2_grid(sta).cpu().numpy()
        ends_gd = self._world_2_grid(ends).cpu().numpy()
        max_range = self.cspace_cfg["cost_ray_max_range"]

        for i in range(self.cost_ray_num_beams):
            frac = max(min(max_d[i].item() / max_range, 1.0), 0.0)
            pygame.draw.aaline(self.canvas,
                               (int(255 * (1 - frac)), int(255 * frac), 0),
                               sta_gd, ends_gd[i])

        # global-plan 路径叠加（品红折线，画在 cost rays 之上）
        pp = getattr(self, "plan_poses", None)
        if pp is not None and pp[self.ri] is not None:
            draw_planned_path(self, pp[self.ri])

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

    def xy_projection_rgb(self, env_idx=0):
        """XY projection of the 3-D c-space as an RGB image (mirrors
        `ConfigurationMapper::getXYProjectionImage`).

        Each cell's colour encodes *which* theta orientation(s) collide there:
        red = small theta (0 .. nz/3), green = middle third, blue = large third.
        Free cells and cells occupied at every theta are both grey (the per-third
        occupancy is balanced / zero); only cells colliding in a subset of theta
        thirds are coloured.  Returns a (G, G, 3) uint8 RGB array in robot-centric
        grid coordinates (robot at the centre, +x = forward).
        """
        cs = self.cspace[env_idx]                            # (G, G, nz) bool
        nz = cs.shape[-1]
        third = nz // 3
        dists = []
        for lo, hi in ((0, third), (third, 2 * third), (2 * third, nz)):
            w = torch.zeros(nz, device=cs.device, dtype=torch.float32)
            w[lo:hi] = 1.0
            dists.append((cs.float() * w).sum(dim=-1))      # (G, G)
        red_d, green_d, blue_d = dists
        avg = (red_d + green_d + blue_d) / 3.0
        rgb = torch.stack(
            [torch.sigmoid(d - avg) for d in (red_d, green_d, blue_d)],
            dim=-1,
        )                                                    # (G, G, 3) in [0,1]
        return (rgb * 255.0).to(torch.uint8)

    # ------------------------------------------------------------- gym-like API
    def reset(self):
        _, info = super().reset()                     # base state discarded
        if self.global_plan:
            self._rebuild_planner()                   # world_occ_full 已更新 -> 重建图
            for i in range(self.N):
                self._replan(i)
        return self._get_cspace_state(current_a=None), info

    def step(self, current_a):
        if self.global_plan:
            self._advance_waypoints()                 # 步首：按 wp_reach 推进航点
        _, reward, dw, tr, info = super().step(current_a)
        if self.global_plan:
            # super().step() 内的 _AutoReset 已给 done env 分配新真目标 -> 重规划
            self._replan_after_reset(dw, tr, None)
        state = self._get_cspace_state(current_a=current_a)
        dw, stuck = self._apply_stuck_reset(dw)
        if stuck is not None and stuck.any():
            state = self._get_cspace_state(current_a=current_a)   # 重置后重算观测
            if self.global_plan:
                self._replan_after_reset(None, None, stuck)       # stuck env 换了位姿+目标
        # 碰撞计数 = 本步因卡死（is_stuck）而触发的重置事件。基类的 point-lidar
        # collide_vec（9cm 圆 + ld_acc 失配）已失效，这里用 stuck-reset 掩码代替，
        # 使「一旦因为碰撞(卡死)而重置，碰撞计数就 +1」；并施加 -200 惩罚。
        self.collide_vec = stuck if stuck is not None else torch.zeros(
            self.N, dtype=torch.bool, device=self.dvc)
        reward = reward.clone()
        reward[self.collide_vec] = self.PUNISH
        return state, reward, dw, tr, info


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
        if self.global_plan:
            self._rebuild_planner()                      # 障碍物重新随机 -> 重建共享图
        return self._get_cspace_state(current_a=None), info

    def _handle_click(self, xy):
        """点击 → 全局规划 → 整条路径入队（global_plan 模式）。

        复写 `Sparrow_PlayGround._handle_click`：把规划出的航点序列整条塞进
        global_path / waypoints，基类 `Sparrow_PlayGround.step` 的 win-pop 机制会
        逐航点推进；点击点（真目标）是最后一个航点。无可行路径时回退基类单点行为。
        """
        if not self.global_plan:
            return super()._handle_click(xy)
        car = self.car_state[self.ri]
        res = self.planner.plan(
            (float(car[0]), float(car[1]), float(car[2])),
            (float(xy[0]), float(xy[1]), None),           # 目标朝向任意
        )
        if res is None or len(res.poses) < 2:
            print("No global path to the clicked point; using a direct waypoint.")
            return super()._handle_click(xy)
        poses = res.poses
        self.plan_poses[self.ri] = poses
        self.plan_idx[self.ri] = 1
        self.global_path.clear()
        self.waypoints.clear()
        for x, y, _t in poses:
            p = np.array([float(x), float(y)], dtype=np.float32)
            self.global_path.append(p)
            self.waypoints.append(p)
        nxt = poses[1]
        self.target_point = torch.tensor([nxt[0], nxt[1]], device=self.dvc)

    def _reset_env(self, i):
        """Stuck reset for the playground — mirrors `Sparrow_PlayGround._AutoReset`:
        new car pose, clear waypoints/global path, target = new pose."""
        init_xy = self._car_loc_init(i)
        self.global_path.clear()
        self.waypoints.clear()
        self.global_path.append(init_xy.cpu().numpy())
        self.waypoints.append(init_xy.cpu().numpy())
        self.target_point = init_xy
        self.step_counter_vec[i] = 0

    def _AutoReset(self):
        """碰撞不再触发重置（去除原 Sparrow_PlayGround 的碰撞重置）。

        卡死由 `_apply_stuck_reset` -> `_reset_env` 处理；到达目标点仅由 dw_vec
        标记 episode 结束，等待鼠标点击新航点继续。
        """
        pass

    def step(self, current_a):
        _, reward, dw, tr, info = Sparrow_PlayGround.step(self, current_a)
        state = self._get_cspace_state(current_a=current_a)
        dw, stuck = self._apply_stuck_reset(dw)
        if stuck is not None and stuck.any():
            state = self._get_cspace_state(current_a=current_a)   # 重置后重算观测
        # 碰撞计数 = 卡死(stuck)重置事件（与 SparrowCspaceEnv.step 一致），驱动 HUD 的 COLLIDE 提示
        self.collide_vec = stuck if stuck is not None else torch.zeros(
            self.N, dtype=torch.bool, device=self.dvc)
        return state, reward, dw, tr, info


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
