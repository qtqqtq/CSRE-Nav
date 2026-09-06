"""
Shared pygame overlay helpers for Sparrow c-space environments.

Used by both `play_cspace.py` (single rendered env, N=1) and
`train.py` (one rendered env out of N parallel envs) to draw the
same debug overlays on top of `Sparrow._render_frame`:

    - draw_hud              : instantaneous / cumulative reward + win/collide flags
    - draw_cost_rays        : the NavRL c-space cost rays the CNN consumes
    - draw_cspace_projection: RGB XY projection of the 3-D c-space
    - draw_lidar_ogm        : robot-centric occupancy grid rasterised from the lidar

Every function reads the rendered environment through `env.ri`, so they work for
both N=1 playgrounds and N>1 vectorized training (where only env `ri` is drawn).
"""
import numpy as np
import pygame


def draw_hud(env, font, reward_val, ep_reward):
    """Overlay the instantaneous / cumulative reward onto the rendered window."""
    if env.window is None:
        return
    lines = [("Reward: %+.2f" % reward_val,
              (120, 255, 120) if reward_val >= 0 else (255, 120, 120)),
             ("Total : %+.2f" % ep_reward, (255, 255, 255))]
    if bool(env.win_vec[env.ri].item()):
        lines.append(("WIN", (120, 255, 120)))
    if bool(env.collide_vec[env.ri].item()):
        lines.append(("COLLIDE", (255, 80, 80)))

    panel = pygame.Surface((200, 18 * len(lines) + 12), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 160))
    env.window.blit(panel, (8, 8))
    y = 12
    for text, color in lines:
        env.window.blit(font.render(text, True, color), (16, y))
        y += 18


def draw_cost_rays(env, font):
    """Overlay the c-space cost rays onto the rendered window.

    Two views of the same data:
      1. on-map boundary -- a cyan polyline + proximity-coloured dots showing,
         for each beam, the nearest footprint-collision distance
         (min over the nz theta layers) in world coordinates.
      2. HUD heatmap      -- the exact B×nz processed cost-ray tensor the CNN
         consumes (near obstacle -> red, free -> green), in the top-right corner.
    """
    if env.window is None or getattr(env, "cost_rays_raw", None) is None:
        return

    raw = env.cost_rays_raw[env.ri].detach().cpu().numpy()   # (B, nz) metres
    min_d = raw.min(axis=1)                                  # (B,) nearest hit per beam
    car = env.car_state[env.ri].detach().cpu().numpy()       # (5,) [x_cm, y_cm, heading, ...]
    x, y, t = float(car[0]), float(car[1]), float(car[2])

    beams = np.arange(env.cost_ray_num_beams) * np.radians(
        env.cspace_cfg["cost_ray_h_res_degrees"])            # robot-frame angles
    world_ang = beams - t                                    # -> world (pygame y-down)
    dx = np.cos(world_ang)
    dy = np.sin(world_ang)
    max_range = env.cspace_cfg["cost_ray_max_range"]         # metres

    end_x = x + min_d * 100.0 * dx                           # world cm
    end_y = y + min_d * 100.0 * dy
    pts = [(int(np.floor(end_x[i])), int(np.floor(end_y[i])))
           for i in range(env.cost_ray_num_beams)]

    if len(pts) > 2:
        pygame.draw.polygon(env.window, (0, 220, 220), pts, 2)
    for i, p in enumerate(pts):
        frac = min(max(min_d[i] / max_range, 0.0), 1.0)
        pygame.draw.circle(env.window, (int(255 * frac), int(255 * (1 - frac)), 0),
                           p, 2)

    # --- HUD heatmap: processed cost-ray input (B x nz) ---
    proc = env.cost_rays_proc[env.ri, 0].detach().cpu().numpy()   # (B, nz), near -> high
    B, L = proc.shape
    cell = 5
    W, H = B * cell, L * cell
    panel = pygame.Surface((W + 4, H + 4 + 16), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 160))
    panel.blit(font.render("cost rays", True, (255, 255, 255)), (4, 2))
    for i in range(B):
        for j in range(L):
            v = min(max(proc[i, j] / max_range, 0.0), 1.0)   # 0 free -> 1 near
            col = (int(255 * v), int(255 * (1 - v)), 40)
            pygame.draw.rect(panel, col, (2 + i * cell, 16 + j * cell, cell, cell))
    env.window.blit(panel, (env.window_size - W - 12, 8))


def draw_cspace_projection(env, font, size=200):
    """Overlay the c-space XY projection (like getXYProjectionImage).

    Colour encodes which theta orientation(s) collide at each cell (red = small
    theta, green = middle, blue = large; grey = free or all theta).  The grid is
    robot-centric with forward up and left left; the robot is the centre dot +
    forward arrow.
    """
    if env.window is None or getattr(env, "cspace", None) is None:
        return
    rgb = env.xy_projection_rgb(env.ri).cpu().numpy()        # (G, G, 3) uint8
    rgb = np.ascontiguousarray(np.rot90(rgb, k=2))           # forward -> up
    G = rgb.shape[0]
    surf = pygame.image.frombuffer(rgb.tobytes(), (G, G), "RGB")
    surf = pygame.transform.scale(surf, (size, size))

    panel = pygame.Surface((size + 4, size + 4 + 16), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 160))
    panel.blit(font.render("c-space XY", True, (255, 255, 255)), (4, 2))
    panel.blit(surf, (2, 18))

    c = 2 + size // 2
    cy = 18 + size // 2
    pygame.draw.circle(panel, (255, 255, 255), (c, cy), 3)
    pygame.draw.line(panel, (0, 255, 255), (c, cy), (c, cy - size // 4), 2)

    env.window.blit(panel, (env.window_size - size - 12,
                            env.window_size - size - 12))


def draw_lidar_ogm(env, font, size=200):
    """Overlay the robot-centric occupancy grid rasterised from the lidar scan.

    Occupied cells come straight from `env.occ_grid` -- the exact grid the
    c-space pipeline builds each step (`_build_occ_grid`, which mirrors the
    Gazebo `lidar_to_occupancy_node`) -- so the panel shows the robot's actual
    line-of-sight perception, not ground truth.  Free cells swept by each beam
    are rasterised here (dim teal) so the grid reads as a local costmap; the
    occupied hit cell is bright; everything out of range stays dark unknown.
    Orientation matches `draw_cspace_projection` (forward up, left left): the
    grid is built with rows = rel_x (forward) and then rot90(k=2) before display.
    """
    if env.window is None or getattr(env, "occ_grid", None) is None:
        return
    occ = env.occ_grid[env.ri].cpu().numpy()            # (G,G) bool occupied
    G = occ.shape[0]
    R = G // 2
    g = env.cspace_cfg["g_min"] * 100.0                 # cm per cell

    car = env.car_state[env.ri].detach().cpu().numpy()  # (5,) [x_cm, y_cm, heading, ...]
    t = float(car[2])
    ld = env.ld_scan_result[env.ri].detach().cpu().numpy()       # (ld_num,) cm
    dirs = env.ld_vectors_wd[env.ri].detach().cpu().numpy()      # (ld_num,2) world unit
    d_range = float(env.ld_d_range)

    # free cells swept along each beam, up to (but excluding) the hit cell.
    # Same robot-frame rotation as `_build_occ_grid`: rel = R(-theta) @ world.
    free = np.zeros((G, G), dtype=bool)
    cos_t, sin_t = np.cos(t), np.sin(t)
    for d, (ux, uy) in zip(ld, dirs):
        rx = ux * cos_t - uy * sin_t                    # robot-frame unit dir
        ry = ux * sin_t + uy * cos_t
        n = min(int(min(d, d_range) // g), G - 1)       # swept cells before the hit
        for s in range(1, n + 1):
            ci = R + int(round(rx * s))
            cj = R + int(round(ry * s))
            if 0 <= ci < G and 0 <= cj < G:
                free[ci, cj] = True

    img = np.zeros((G, G, 3), dtype=np.uint8)
    img[free] = (40, 60, 70)                            # swept free: dim teal
    img[occ] = (220, 60, 40)                            # occupied hit: bright red-orange
    rgb = np.ascontiguousarray(np.rot90(img, k=2))      # forward -> up
    surf = pygame.image.frombuffer(rgb.tobytes(), (G, G), "RGB")
    surf = pygame.transform.scale(surf, (size, size))

    panel = pygame.Surface((size + 4, size + 4 + 16), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 160))
    panel.blit(font.render("lidar OGM", True, (255, 255, 255)), (4, 2))
    panel.blit(surf, (2, 18))

    c = 2 + size // 2
    cy = 18 + size // 2
    pygame.draw.circle(panel, (255, 255, 255), (c, cy), 3)
    pygame.draw.line(panel, (0, 255, 255), (c, cy), (c, cy - size // 4), 2)

    env.window.blit(panel, (12, env.window_size - size - 12))
