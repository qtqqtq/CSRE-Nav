"""
Interactive test script: drive a trained CNNSAC (c-space) model with the mouse.

Mimics `ColorDynamic/play_with_mouse.py`: click the rendered map to add target
waypoints, press Esc to reset, close the window to quit.  Loads a model from
`pytorch_models/` and runs the policy in a rendered `SparrowCspacePlayground`.

Usage (inside the container, with a display / GPU):

    python3 play_cspace.py --ModelName SAC_sparrow_cspace --dvc cuda

The model consumes the
(RAY_CHANNELS×RAY_H_BEAMS×RAY_V_LAYERS + KINEMATIC_DIM)-dim c-space state
(by default 3×36×30 + 4 = 3244 ray-segment grid +
[dist, heading, normalized actual v, normalized actual w]), so it must have been
trained on `SparrowCspaceEnv` (e.g. `SAC_sparrow_cspace`).

Clicking the map adds a local waypoint within the configured distance limit.

On top of the overlays from `render_utils`, `draw_interval_rays` (on by default,
disable with `--interval_rays False`) draws the interval-ray far boundary: a cyan
polyline at each beam's nearest collision (min `entry` over the theta layers) and
one orange ray per colliding beam out to its farthest `exit` radius.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pygame
import torch

import settings
from Sparrow_V2 import str2bool, Sparrow, SparrowCspacePlayground
from cnn_network import CNNSAC
from render_utils import (draw_hud, draw_cost_rays, draw_interval_rays,
                          draw_cspace_projection, draw_lidar_ogm)


parser = argparse.ArgumentParser()

# ---- Model / agent ----
parser.add_argument('--ModelName', type=str, default='interval_cspace_2',
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
                    help='number of moving obstacles per env')
parser.add_argument('--Obs_V', type=int, default=settings.SPARROW_OBS_V,
                    help='max moving-obstacle velocity (cm/s)')
parser.add_argument('--MapObs', type=str, default=None, help="map file, e.g. 'map.png' or None")
parser.add_argument('--render_speed', type=str, default=settings.SPARROW_RENDER_SPEED,
                    help='fast / slow / real (slow = 30 FPS, stable display)')
parser.add_argument('--max_ep_steps', type=int, default=500)

# ---- Lidar scan (mirrors ColorDynamic/play_with_mouse.py defaults) ----
parser.add_argument('--ld_a_range', type=int, default=360,
                    help='max scanning angle of lidar (degree)')
parser.add_argument('--ld_d_range', type=int,
                    default=int(round(settings.CSPACE_COST_RAY_MAX_RANGE * 100)),
                    help='max lidar range (cm); defaults to the training value, '
                         'which is exactly the c-space observation range')
parser.add_argument('--ld_num', type=int, default=36,
                    help='number of lidar rays per world (training default; '
                         'the c-space observation is built from these hits)')
parser.add_argument('--ld_GN', type=int, default=3,
                    help='lidar rays grouped (down-sampled) per group')
parser.add_argument('--ld_acc', type=int, default=settings.SPARROW_LD_ACC,
                    help='lidar DDA step size in cm (training default; the '
                         'default 3 cm here used to disagree with training)')
parser.add_argument('--backend', choices=('grid', 'points'),
                    default=settings.CSPACE_BACKEND,
                    help='c-space observation geometry: grid (raster) or '
                         'points (analytic lidar point set)')

# ---- Stuck auto-reset (cost rays ~0 in every direction) ----
parser.add_argument('--stuck_reset', type=str2bool, default=settings.SPARROW_STUCK_RESET,
                    help='auto-reset the episode when the robot is boxed in')
parser.add_argument('--stuck_threshold', type=float, default=0.1,
                    help='cost-ray distance (m) below which a beam counts as jammed '
                         '(play default 0.5; training default is settings.SPARROW_STUCK_THRESHOLD)')

# ---- Interval-ray overlay (entry polyline + exit rays) ----
parser.add_argument('--interval_rays', type=str2bool, default=True,
                    help='overlay the interval rays: cyan polyline = nearest entry '
                         '(min over theta), orange ray = farthest exit (max over the '
                         'colliding theta layers)')

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
        ld_acc=opt.ld_acc,
        noise=False,
        DR=False,
        stuck_reset=opt.stuck_reset,
        stuck_threshold=opt.stuck_threshold,
        random_footprint=settings.RANDOM_FOOTPRINT_ENABLED,
        cspace_cfg={"backend": opt.backend},
    )

    network = CNNSAC(
        kinematic_dim=settings.KINEMATIC_DIM,
        action_dim=settings.ACTION_DIM,
        max_action=settings.MAX_ACTION,
        h_beams=settings.RAY_H_BEAMS,
        v_layers=settings.RAY_V_LAYERS,
        ray_channels=settings.RAY_CHANNELS,
        cnn_feature_dim=64,
        lr=settings.LR,
        actor_lr=settings.ACTOR_LR,
        critic_lr=settings.CRITIC_LR,
        initial_alpha=settings.SAC_INITIAL_ALPHA,
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

        # unpack flat state -> (1,C,H,V) interval rays + (1,4) kinematic
        rg = s[:, :settings.RAY_GRID_FLAT_DIM].reshape(
            1, settings.RAY_CHANNELS,
            settings.RAY_H_BEAMS, settings.RAY_V_LAYERS)
        kin = s[:, settings.RAY_GRID_FLAT_DIM:]

        with torch.no_grad():
            if opt.deterministic:
                action = network.actor.get_mean_action(rg, kin)   # (1,2) in [-1,1]
            else:
                action, _ = network.actor.sample(rg, kin)

            # 当前执行动作的 Q 估计(双 critic): 与训练约定一致 —— 喂原始 tanh(-1,1) 动作,
            # 而非映射后的 a_in(env 真正执行的物理速度)。min-Q 是 SAC 优化的目标。
            q1, q2 = network.critic(rg, kin, action)
            q_pair = (float(q1.item()), float(q2.item()))

        # same action convention as training: linear a0 in [-1,1] if ALLOW_REVERSE
        a_in = settings.map_continuous_action(action)
        s, r, dw, tr, info = env.step(a_in)

        # PlayGround 的 reward 恒为 0；用训练用的完整奖励函数实时重算并显示
        Sparrow._reward_function(env, a_in)
        reward_val = float(env.reward_vec.item())
        ep_reward += reward_val
        draw_cost_rays(env, font)
        if opt.interval_rays:
            draw_interval_rays(env, font)
        draw_cspace_projection(env, font)
        draw_lidar_ogm(env, font)
        draw_hud(env, font, reward_val, ep_reward, q_pair=q_pair)
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
