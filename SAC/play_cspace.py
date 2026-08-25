"""
Interactive test script: drive a trained CNNSAC (c-space) model with the mouse.

Mimics `ColorDynamic/play_with_mouse.py`: click the rendered map to add target
waypoints, press Esc to reset, close the window to quit.  Loads a model from
`pytorch_models/` and runs the policy in a rendered `SparrowCspacePlayground`.

Usage (inside the container, with a display / GPU):

    python3 play_cspace.py --ModelName SAC_sparrow_cspace --dvc cuda
    python3 play_cspace.py --ModelName SAC_sparrow_cspace --dvc cuda --global_plan

The model consumes the 364-dim c-space state (360 cost rays + 4 kinematic), so
it must have been trained on `SparrowCspaceEnv` (e.g. `SAC_sparrow_cspace`).

With `--global_plan`, clicking the map plans a global path in the c-space graph
(`GraphPlanner`) from the robot to the clicked point and enqueues its waypoints;
without it, a click adds a single local waypoint as before.
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
from render_utils import draw_hud, draw_cost_rays, draw_cspace_projection


parser = argparse.ArgumentParser()

# ---- Model / agent ----
parser.add_argument('--ModelName', type=str, default='SAC_sparrow_cspace_1',
                    help='model basename in pytorch_models (e.g. SAC_sparrow_cspace)')
parser.add_argument('--ModelDir', type=str, default=settings.MODEL_DIR,
                    help='directory holding *_actor.pth / *_critic.pth')
parser.add_argument('--deterministic', type=str2bool, default=True,
                    help='True: mean action (eval); False: sampled action')

# ---- Sparrow env ----
parser.add_argument('--dvc', type=str, default='cuda', help='cuda / cpu')
parser.add_argument('--window_size', type=int, default=800)
parser.add_argument('--D', type=int, default=400, help='max local planning distance')
parser.add_argument('--O', type=int, default=settings.SPARROW_N_OBSTACLES,
                    help='number of obstacles per env')
parser.add_argument('--Obs_V', type=int, default=30, help='max obstacle velocity (cm/s)')
parser.add_argument('--MapObs', type=str, default=None, help="map file, e.g. 'map.png' or None")
parser.add_argument('--render_speed', type=str, default=settings.SPARROW_RENDER_SPEED,
                    help='fast / slow / real (slow = 30 FPS, stable display)')
parser.add_argument('--max_ep_steps', type=int, default=500)

# ---- Lidar scan (mirrors ColorDynamic/play_with_mouse.py defaults) ----
parser.add_argument('--ld_a_range', type=int, default=360,
                    help='max scanning angle of lidar (degree)')
parser.add_argument('--ld_d_range', type=int, default=400,
                    help='max lidar range (cm); should cover the c-space window (4 m)')
parser.add_argument('--ld_num', type=int, default=72,
                    help='number of lidar rays per world')
parser.add_argument('--ld_GN', type=int, default=3,
                    help='lidar rays grouped (down-sampled) per group')

# ---- Stuck auto-reset (cost rays ~0 in every direction) ----
parser.add_argument('--stuck_reset', type=str2bool, default=settings.SPARROW_STUCK_RESET,
                    help='auto-reset the episode when the robot is boxed in')
parser.add_argument('--stuck_threshold', type=float, default=0.5,
                    help='cost-ray distance (m) below which a beam counts as jammed '
                         '(play default 0.5; training default is settings.SPARROW_STUCK_THRESHOLD)')

# ---- Global planner (click -> plan a path in c-space, follow waypoints) ----
parser.add_argument('--global_plan', action='store_true',
                    default=settings.GLOBAL_PLAN_ENABLED,
                    help='enable global graph planning: clicking the map plans a full '
                         'path from the robot to the clicked point and enqueues its '
                         'waypoints (else: single-waypoint click behavior)')

opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc)


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
        stuck_reset=opt.stuck_reset,
        stuck_threshold=opt.stuck_threshold,
        global_plan=opt.global_plan,
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
        draw_cost_rays(env, font)
        draw_cspace_projection(env, font)
        draw_hud(env, font, reward_val, ep_reward)
        pygame.display.update()   # 每帧在所有叠加层绘制完成后只 flush 一次，避免闪动
        if bool(dw.item()):   # win / collide / stuck -> 一个 episode 结束，累计奖励清零
            ep_reward = 0.0

        # handle window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return


if __name__ == '__main__':
    main()
