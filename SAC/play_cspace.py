"""
Interactive test script: drive a trained CNNSAC (c-space) model with the mouse.

Mimics `ColorDynamic/play_with_mouse.py`: click the rendered map to add target
waypoints, press Esc to reset, close the window to quit.  Loads a model from
`pytorch_models/` and runs the policy in a rendered `SparrowCspacePlayground`.

Usage (inside the container, with a display / GPU):

    python3 play_cspace.py --ModelName SAC_sparrow_cspace --dvc cuda

The model consumes the 364-dim c-space state (360 cost rays + 4 kinematic), so
it must have been trained on `SparrowCspaceEnv` (e.g. `SAC_sparrow_cspace`).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pygame
import torch

import settings
from Sparrow_V2 import str2bool, Sparrow
from sparrow_cspace_env import SparrowCspacePlayground
from cnn_network import CNNSAC


parser = argparse.ArgumentParser()

# ---- Model / agent ----
parser.add_argument('--ModelName', type=str, default='SAC_sparrow_cspace',
                    help='model basename in pytorch_models (e.g. SAC_sparrow_cspace)')
parser.add_argument('--ModelDir', type=str, default=settings.MODEL_DIR,
                    help='directory holding *_actor.pth / *_critic.pth')
parser.add_argument('--deterministic', type=str2bool, default=True,
                    help='True: mean action (eval); False: sampled action')

# ---- Sparrow env ----
parser.add_argument('--dvc', type=str, default='cuda', help='cuda / cpu')
parser.add_argument('--window_size', type=int, default=800)
parser.add_argument('--D', type=int, default=400, help='max local planning distance')
parser.add_argument('--O', type=int, default=15, help='number of obstacles per env')
parser.add_argument('--Obs_V', type=int, default=30, help='max obstacle velocity (cm/s)')
parser.add_argument('--MapObs', type=str, default=None, help="map file, e.g. 'map.png' or None")
parser.add_argument('--render_speed', type=str, default='slow',
                    help='fast / slow / real (slow = 30 FPS, stable display)')
parser.add_argument('--max_ep_steps', type=int, default=500)

# ---- Lidar scan (mirrors ColorDynamic/play_with_mouse.py defaults) ----
parser.add_argument('--ld_a_range', type=int, default=360,
                    help='max scanning angle of lidar (degree)')
parser.add_argument('--ld_d_range', type=int, default=300,
                    help='max scanning distance of lidar (cm)')
parser.add_argument('--ld_num', type=int, default=72,
                    help='number of lidar rays per world')
parser.add_argument('--ld_GN', type=int, default=3,
                    help='lidar rays grouped (down-sampled) per group')

# ---- Stuck auto-reset (cost rays ~0 in every direction) ----
parser.add_argument('--stuck_reset', type=str2bool, default=True,
                    help='auto-reset the episode when the robot is boxed in')
parser.add_argument('--stuck_threshold', type=float, default=0.15,
                    help='cost-ray distance (m) below which a beam counts as jammed')

opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc)


def _draw_hud(env, font, reward_val, ep_reward):
    """Overlay the instantaneous / cumulative reward onto the rendered window."""
    if env.window is None:
        return
    lines = [("Reward: %+.2f" % reward_val,
              (120, 255, 120) if reward_val >= 0 else (255, 120, 120)),
             ("Total : %+.2f" % ep_reward, (255, 255, 255))]
    if bool(env.win_vec.item()):
        lines.append(("WIN", (120, 255, 120)))
    if bool(env.collide_vec.item()):
        lines.append(("COLLIDE", (255, 80, 80)))

    panel = pygame.Surface((200, 18 * len(lines) + 12), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 160))
    env.window.blit(panel, (8, 8))
    y = 12
    for text, color in lines:
        env.window.blit(font.render(text, True, color), (16, y))
        y += 18
    pygame.display.update()


def _draw_cost_rays(env, font):
    """Overlay the c-space cost rays onto the rendered window.

    Two views of the same data:
      1. on-map boundary -- a cyan polyline + proximity-coloured dots showing,
         for each of the 36 beams, the nearest footprint-collision distance
         (min over the 10 theta layers) in world coordinates.
      2. HUD heatmap      -- the exact 36x10 processed cost-ray tensor the CNN
         consumes (near obstacle -> red, free -> green), in the top-right corner.
    """
    if env.window is None or getattr(env, "cost_rays_raw", None) is None:
        return

    raw = env.cost_rays_raw[0].detach().cpu().numpy()   # (36, 10) metres
    min_d = raw.min(axis=1)                              # (36,) nearest hit per beam
    car = env.car_state[env.ri].detach().cpu().numpy()   # (5,) [x_cm, y_cm, heading, ...]
    x, y, t = float(car[0]), float(car[1]), float(car[2])

    beams = np.arange(env.cost_ray_num_beams) * np.radians(
        env.cspace_cfg["cost_ray_h_res_degrees"])        # robot-frame angles
    world_ang = beams - t                                # -> world (pygame y-down)
    dx = np.cos(world_ang)
    dy = np.sin(world_ang)
    max_range = env.cspace_cfg["cost_ray_max_range"]     # metres

    end_x = x + min_d * 100.0 * dx                       # world cm
    end_y = y + min_d * 100.0 * dy
    pts = [(int(np.floor(end_x[i])), int(np.floor(end_y[i])))
           for i in range(env.cost_ray_num_beams)]

    if len(pts) > 2:
        pygame.draw.polygon(env.window, (0, 220, 220), pts, 2)
    for i, p in enumerate(pts):
        frac = min(max(min_d[i] / max_range, 0.0), 1.0)
        pygame.draw.circle(env.window, (int(255 * frac), int(255 * (1 - frac)), 0),
                           p, 2)

    # --- HUD heatmap: processed cost-ray input (36 x 10) ---
    proc = env.cost_rays_proc[0, 0].detach().cpu().numpy()   # (36, 10), near -> high
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


def main():
    env = SparrowCspacePlayground(
        N=1,
        O=opt.O,
        dvc=opt.dvc,
        action_type='Continuous',
        render_mode='human',
        render_speed=opt.render_speed,
        window_size=opt.window_size,
        D=opt.D,
        MapObs=opt.MapObs,
        Obs_V=opt.Obs_V,
        max_ep_steps=opt.max_ep_steps,
        ld_a_range=opt.ld_a_range,
        ld_d_range=opt.ld_d_range,
        ld_num=opt.ld_num,
        ld_GN=opt.ld_GN,
        noise=False,
        DR=False,
    )

    network = CNNSAC(
        kinematic_dim=settings.KINEMATIC_DIM,
        action_dim=settings.ACTION_DIM,
        max_action=settings.MAX_ACTION,
        h_beams=settings.RAY_H_BEAMS,
        v_layers=settings.RAY_V_LAYERS,
        cnn_feature_dim=128,
        lr=settings.LR,
        device=opt.dvc,
    )
    network.load(opt.ModelName, opt.ModelDir)
    print("Loaded model: %s" % opt.ModelName)
    print("Click the map to add a waypoint. Press Esc to reset. Close window to quit.")

    pygame.font.init()
    font = pygame.font.Font(None, 26)

    s, info = env.reset()
    ep_reward = 0.0
    while True:
        # Esc -> reset environment (and clear the waypoint list)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            s, info = env.reset()
            ep_reward = 0.0

        # unpack the flat c-space state -> (1,1,H,V) ray grid + (1,4) kinematic
        rg = s[:, :settings.RAY_GRID_FLAT_DIM].reshape(
            1, 1, settings.RAY_H_BEAMS, settings.RAY_V_LAYERS)
        kin = s[:, settings.RAY_GRID_FLAT_DIM:]

        with torch.no_grad():
            if opt.deterministic:
                action = network.actor.get_mean_action(rg, kin)   # (1,2) in [-1,1]
            else:
                action, _ = network.actor.sample(rg, kin)

        # same action convention as training: linear in [0,1], angular in [-1,1]
        a_in = torch.stack([(action[:, 0] + 1) / 2, action[:, 1]], dim=1)
        s, r, dw, tr, info = env.step(a_in)

        # PlayGround 的 reward 恒为 0；用训练用的完整奖励函数实时重算并显示
        Sparrow._reward_function(env, a_in)
        reward_val = float(env.reward_vec.item())
        ep_reward += reward_val
        _draw_cost_rays(env, font)
        _draw_hud(env, font, reward_val, ep_reward)
        if bool(dw.item()):   # win / collide -> 一个 episode 结束，累计奖励清零
            ep_reward = 0.0

        # cost rays ~0 in every direction -> robot wedged; force a reset
        if opt.stuck_reset and bool(env.is_stuck(opt.stuck_threshold).item()):
            print("stuck (cost rays ~0), resetting")
            s, info = env.reset()
            ep_reward = 0.0

        # handle window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return


if __name__ == '__main__':
    main()
