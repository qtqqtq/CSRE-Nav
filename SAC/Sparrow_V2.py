from collections import deque
from scipy import ndimage
import numpy as np
import torch
import copy
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame


'''
Sparrow-V2: A Reinforcement Learning Friendly Mobile Robot Simulator for Dynamic Environments
Author: Jinghao Xin (https://github.com/XinJingHao)
Date: 2025/01/05
License: GPL-3.0

Good features:
· Vectorizable (Super fast data collection)
· High Diversity (Randomly-generated static obstacles, Randomly moving dynamic obstacles)
· Easy Sim2Real (Domain randomization and sensor noise supported)
· Lightweight (Consume a few GPU memories for vectorized environments)
· Standard Gym API with Pytorch data flow
· GPU/CPU are both acceptable (If you use Pytorch to build your RL model, you can run your RL model and Sparrow both on GPU. Then you don't need to transfer the data from CPU to GPU anymore.)
· Easy to use (Pure Python files. Just import, never worry about installation)
· Ubuntu/Windows/MacOS are both supported
· Accept image as map (Customize your own environments easily and rapidly)


“The sparrow may be small but it has all the vital organs.”
“麻雀虽小，五脏俱全。”


Only for non-commercial purposes.
All rights reserved. 
'''


# 19 editable configurations:
default_cfg = dict(
    dvc=torch.device('cpu'), # running device of Sparrow: cuda / cpu
    action_type='Discrete', # 'Discrete' / 'Continuous' action
    window_size=800, # size of the training map
    D=400, # maximal local planning distance
    N=1, # number of vectorized environments
    O=5, # number of dynamic (moving) obstacles in each environment
    dynamic_obs=True,  # whether to spawn moving (dynamic) obstacles at all
    RdON=False,  # whether to randomize(decrease) the number of obstacles in vectorized envs
    ScOV=False,  # whether to scale the maximal velocity of obstacles of each env copy
    RdOV=True,  # whether to randomize the velocity of obstacles
    RdOT=True, # whether to randomize the type of obstacles
    RdOR=True,  # whether to generate obstacles with random radius between [10, Obs_R]
    Obs_R=14,  # maximal obstacle radius, cm
    Obs_V=25,  # maximal obstacle velocity, cm/s
    generate_rate=0.5,  # per-region probability of spawning a static obstacle
    ang_vel_penalty=0.3,  # penalty weight on |angular velocity| (discourage spinning in place)
    stall_penalty=0.6,    # |实际线速度| < stall_threshold 时的惩罚(满额=静止, 线性过渡)。
                          # 原地逗留与"前后抖动"(高频正负指令经 K 滤波后 |v| 起不来)都由该项收口；
                          # 匀速前进/倒车(低通后 |v| 足够大) R_stall=0，不罚。倒车方向正误交给 R_distance。
    stall_threshold=0.2,  # 线速度死区阈值, 以 v_linear_max(=50 cm/s) 的比例计(0.2 = 10 cm/s)
    MapObs=None, # None /or/ the name of .png file, e.g. None / 'map.png'
    ld_a_range=360, # max scanning angle of lidar (degree)
    ld_d_range=100, # max scanning distance of lidar (cm)
    ld_num=36, # number of lidar streams in each world
    ld_GN=3, # how many lidar streams are grouped(down-sampling) for each group
    ld_acc=3, # lidar scan accuracy / DDA step size (cm); larger = faster but coarser
    ri=0, # render index: the index of world that to be rendered
    basic_ctrl_interval=0.1, # control interval (s), 0.1 means 10 Hz control frequency
    ctrl_delay=0, # control delay, in basic_ctrl_interval, 0 means no control delay
    K=(0.55,0.6), # K_linear, K_angular
    draw_auxiliary = False, # whether to draw auxiliary area
    render_mode='human', # 'human' /or/ None
    render_speed='fast', # 'real' / 'fast' / 'slow'
    max_ep_steps=500, # maximum episodic steps
    noise=False, # whether to add noise to the observations
    DR=False, # whether to use Domain Randomization
    DR_freq=int(3.2e3), # frequency of re-Domain Randomization, in total steps
    compile=False) # whether to torch.compile to boost simulation speed


class Sparrow():
    def __init__(self, **params):
        self.__dict__.update(**default_cfg) # Apply default configuration first
        self.__dict__.update(params)        # Then override with user's configuration
        if self.DR: self.noise = True  # 开启Domain Randomization后默认开启state noise
        self.version = "V2.0"

        '''State/Action/Reward initialization'''
        assert self.ld_num % self.ld_GN == 0  # ld_num must be divisible by ld_GN
        self.grouped_ld_num = int(self.ld_num / self.ld_GN)
        self.absolute_state_dim = 5 + self.grouped_ld_num  # [dx,dy,orientation,v_linear,v_angular] + [lidar result]
        self.state_dim = 8 + self.grouped_ld_num  # [cAl,cAa,rAl,rAa,D2T,alpha,v_linear,v_angular] + [lidar result]
        if self.action_type == 'Discrete': self.action_dim = 7 # 5：前进+转弯,6：前进+转弯+后退,7：前进+转弯+后退+减速,8：前进+转弯+后退+减速+原地静止
        else: self.action_dim = 2 # [V_linear, V_angular]
        self.AWARD, self.PUNISH = 200, -200 # 到达奖励，碰撞、越界惩罚

        '''Car initialization'''
        self.car_radius = 9  # cm
        self.collision_trsd = self.car_radius + 5  # collision threshould, in cm
        self.v_linear_max = 50 # max linear velocity, in cm/s
        self.v_angular_max = 2 # max angular velocity, in rad/s
        if self.action_type == 'Continuous': self.continuous_scale = torch.tensor([[self.v_linear_max, self.v_angular_max]], device=self.dvc) # (1,2)
        self.a_space = torch.tensor([[0.2*self.v_linear_max , self.v_angular_max],[self.v_linear_max , self.v_angular_max],
                                     [self.v_linear_max, 0], # v_linear, v_angular
                                     [self.v_linear_max, -self.v_angular_max],[0.2*self.v_linear_max, -self.v_angular_max],
                                     [-self.v_linear_max, 0],  # -v_linear, v_angular
                                     [0.1*self.v_linear_max, 0], # slow down
                                     [0., 0.]], device=self.dvc) # stop
        self.a_space = self.a_space.unsqueeze(dim=0).repeat((self.N, 1, 1)) # (action_dim,2) -> (N,action_dim,2)
        self.a_state = torch.tensor([[0.2,1], [1,1], [1,0], [1,-1], [0.2,-1], [-1,0], [0.1,0], [0,0]], device=self.dvc) # (action_dim,2)
        if self.action_type == 'Discrete': self.init_pre_action = (self.action_dim-1)*torch.ones(self.N, dtype=torch.int64, device=self.dvc)
        else: self.init_pre_action = torch.zeros((self.N,2), device=self.dvc) # the pre_action of init state
        self.arange_constant = torch.arange(self.N, device=self.dvc) # 仅用于索引
        self.K = torch.tensor([self.K], device=self.dvc) # K_linear, K_angular
        self.ctrl_interval = self.basic_ctrl_interval * torch.ones((self.N,1), device=self.dvc) # control interval, in second; (N,1)
        self.ctrl_pipe_init = deque() # holding the delayed action
        for i in range(self.ctrl_delay): self.ctrl_pipe_init.append(self.init_pre_action) # 控制指令管道初始化


        '''Map initialization'''
        self.obs_canvas_torch = torch.ones((self.window_size, self.window_size), dtype=torch.int64) # 用于将障碍物坐标转化为2D栅格图

        # 占用图像缓冲：把 静态轮廓 ∪ 地图填充 栅格化为一张 1cm 分辨率 bool 图，
        # 供 _ld_scan_vec 的 DDA 命中检测直接查表（替代 O(N*ld_num*B) 的全障碍物集合比对）。
        # base_occ_flat 常驻 dvc，只允许 .copy_() 原地更新（不能重新绑定），否则 torch.compile 包装会失效。
        self.w2 = self.window_size * self.window_size          # 编码值上界 window_size²
        self.base_occ_flat = torch.zeros(self.w2, dtype=torch.bool, device=self.dvc)  # (w2,) bool, C序平坦 idx = x*window_size+y
        self.static_contour_mask = None    # (window,window) np.bool, _static_obstacle_init 里每次 reset 重建
        self.map_filled_mask = None        # (window,window) np.bool, _map_obstacle_init 里建一次（地图静态）
        self.target_area = 30  # # enter the central circle (radius=target_area) will be considered win.
        self.R_map = int(self.window_size/2)
        self.target_point = torch.tensor([[self.R_map,self.R_map]]).repeat(self.N,1).to(self.dvc) # (N,2)

        # Dynamic Obstacle Related:
        self.w = 6  # 动态障碍物线条粗细
        self.b_kernel = np.ones((self.w + 3, self.w + 3)) # 用于腐蚀的卷积核
        self.Obs_refresh_interval = 1 # 障碍物运动刷新周期(相对于小车控制而言，1表示与小车同频率刷新); 增大时,有助于增加Obs速度分辨率,但不利于时序感知
        self.Obs_refresh_counter = self.Obs_refresh_interval
        self.max_Obs_V = int(self.Obs_V * self.Obs_refresh_interval * self.basic_ctrl_interval) # 障碍物x,y轴最大速度 (cm per fresh), 标量
        self.Dynamic_obs_canvas = pygame.Surface((2 * (self.Obs_R + self.w), 2 * (self.Obs_R + self.w)))  # 用于画动态障碍物，方便后续转化为栅格点
        self.l_margin = 50 # left&up margin of dynamic obstacle's moving space
        self.h_margin = self.window_size-50 # right&bottom margin of dynamic obstacle's moving space

        # Static Obstacle Related:
        self.Static_obs_canvas = pygame.Surface((self.window_size, self.window_size))  # 用于画静态障碍物，方便后续转化为栅格点
        self.area = 6 # 横/纵向被切分区域的数量
        self.sn = 2  # 每个矩形区域的障碍物的最大数量
        # self.generate_rate 已在 default_cfg 中定义(每个区域产生静态障碍物的概率)
        self.d_rect = int(self.window_size/self.area) # 生成静态障碍物的矩形区域的边长
        self.rect_offsets = torch.cartesian_prod(torch.arange(0, self.area), torch.arange(0, self.area)).numpy() * self.d_rect

        # Map Obstacle Related:
        if self.MapObs:
            # self.MapObs should be the name of a '.png' file, e.g. self.MapObs = 'map.png'
            # 'map.png' should be of shape (window_size,window_size,3), where 0 represents obstacles and 255 represents free space.
            self._map_obstacle_init()


        '''Lidar initialization'''
        # ld_acc is now a default_cfg parameter (see above); no hardcode here.
        self.ld_scan_result = torch.zeros((self.N, self.ld_num), device=self.dvc)  # used to hold lidar scan result, (N, ld_num)
        self.ld_result_grouped = torch.zeros((self.N, self.grouped_ld_num), device=self.dvc) # the grouped lidar scan result, (N, grouped_ld_num)
        self.ld_angle_interval = torch.arange(self.ld_num, device=self.dvc) * (self.ld_a_range/180) * torch.pi / (self.ld_num) - (self.ld_a_range/360) * torch.pi #(ld_num, )
        self.ld_angle_interval = self.ld_angle_interval.unsqueeze(dim=0).repeat((self.N, 1)) # (N, ld_num)

        '''State noise initialization (unormalized magnitude)'''
        if self.noise:
            self.noise_magnitude = torch.hstack((torch.tensor([2,2,torch.pi/50,1,torch.pi/50]), torch.ones(self.grouped_ld_num))).to(self.dvc) #(23,)

        '''Domain Randomization initialization'''
        if self.DR:
            # 创建基准值，后续在基准值上随机化
            self.ctrl_interval_base = self.ctrl_interval.clone() # (N,1)
            self.K_base = self.K.clone() # (1,2)
            self.a_space_base = self.a_space.clone() # (N,A,2)
            self.noise_magnitude_base = self.noise_magnitude.clone() # (23,)

        '''Pygame initialization'''
        assert self.render_mode is None or self.render_mode == "human"
        # "human": render with pygame
        # None: not render anything
        self.window = None
        self.clock = None
        self.canvas = None
        self.render_rate = self.ctrl_interval[self.ri].item()  # FPS = 1/self.render_rate

        '''Internal variables initialization'''
        # 提前声明变量的数据格式，速度更快
        self.step_counter_DR = 0 # 用于记录DR的持续步数
        self.step_counter_vec = torch.zeros(self.N, dtype=torch.long, device=self.dvc) # 用于truncate
        self.car_state = torch.zeros((self.N, 5), device=self.dvc, dtype=torch.float32)
        self.reward_vec = torch.zeros(self.N, device=self.dvc) # vectorized reward signal
        self.dw_vec = torch.zeros(self.N, dtype=torch.bool, device=self.dvc) # vectorized terminated signal
        self.tr_vec = torch.zeros(self.N, dtype=torch.bool, device=self.dvc)  # vectorized truncated signal
        self.done_vec = torch.zeros(self.N, dtype=torch.bool, device=self.dvc)  # vectorized done signal
        # for state normalization:
        self.state_upperbound = torch.ones(self.state_dim-4, device=self.dvc) # -4 for exclusion of act_state
        self.state_upperbound[0] *= self.D
        self.state_upperbound[1] *= 1 # 仅用于补位,后面单独对alpha归一化
        self.state_upperbound[2] *= self.v_linear_max
        self.state_upperbound[3] *= self.v_angular_max
        self.state_upperbound[4:self.state_dim] *= self.ld_d_range

        '''Logging'''
        if self.dvc.type == 'cpu':
            print("Although Sparrow can be deployed on CPU, we strongly recommend you use GPU to accelerate simulation! "
                  "Please try to use ' dvc=torch.device('cuda') ' when instantiate Sparrow.")
        else:
            # 编译雷达扫描函数，加速仿真. 但有些显卡上会报错.
            if self.compile == True:
                self._ld_scan_vec = torch.compile(self._ld_scan_vec)
            else:
                print("When instantiate Sparrow, you can set 'compile=True' to boost the simulation speed. ")
        print(f"Sparrow-{self.version}, N={self.N}, State dimension={self.state_dim}, {self.action_type} action dimension={self.action_dim}.")

    def _random_noise(self, magnitude:float, size:tuple, device:torch.device):
        '''Generate uniform random noise in [-magnitude,magnitude)'''
        return (torch.rand(size=size, device=device)-0.5) * 2 * magnitude

    def _world_2_grid(self, coordinate_wd):
        ''' Convert world coordinates (denoted by _wd, continuous, unit: cm) to grid coordinates (denoted by _gd, discrete, 1 grid = 1 cm)
            Input: torch.tensor; Output: torch.tensor; Shape: Any shape '''
        return coordinate_wd.floor().int()

    def _build_occ_image(self):
        '''把 静态轮廓 ∪ 地图填充 栅格化为一张 1cm bool 占用图，供 _ld_scan_vec 查表。

        约束: 栅格化用的像素集合必须与 vec_static_bound_code(+vec_map_bound_code)
        逐像素一致，才能保证 DDA 命中检测 bit-exact。两种掩码形状均为
        (window_size, window_size)，C序 reshape(-1) 的平坦下标恰好等于现在的
        编码公式 x*window_size+y（轴0为x、轴1为y、且行宽==window_size）。
        注意用的是*轮廓*static_contour_mask / *填充*map_filled_mask（地图本来就
        按整块 np.where(map_np==0) 编码），不是全局规划器用的填充位图 world_occ_full。
        base_occ_flat 只原地 .copy_()，保持张量身份稳定（torch.compile 安全）。
        '''
        assert self.static_contour_mask is not None, "_build_occ_image() 须在 _static_obstacle_init 之后调用"
        assert self.static_contour_mask.shape == (self.window_size, self.window_size), \
            f"static_contour_mask 形状 {self.static_contour_mask.shape} != ({self.window_size},{self.window_size})"
        occ = self.static_contour_mask
        if self.MapObs:
            assert self.map_filled_mask is not None and \
                self.map_filled_mask.shape == (self.window_size, self.window_size), "map_filled_mask 缺失或形状不符"
            occ = occ | self.map_filled_mask
        flat = occ.reshape(-1)  # C序: flat idx = x*window_size + y, 与代码公式一致
        self.base_occ_flat.copy_(torch.as_tensor(flat, dtype=torch.bool, device=self.dvc))

    def _Domain_Randomization(self):
        # 1) randomize the control interval; ctrl_interval.shape: (N,1)
        self.ctrl_interval = self.ctrl_interval_base + self._random_noise(0.01, (self.N,1), self.dvc)# control interval, in second

        # 2) randomize the kinematic parameter; K.shape: (N,2)
        self.K = self.K_base + self._random_noise(0.05, (self.N,2), self.dvc)# control interval, in second;

        # 3) randomize the max velocity; a_space.shape: (N,6,2)
        self.a_space = self.a_space_base * (1 + self._random_noise(0.05, (self.N, 1, 2), self.dvc)) # Random the maximal speed of each env copy by 0.9~1.1

        # 4) randomize the magnitude of state noise; noise_magnitude.shape: (N,absolute_state_dim)
        self.noise_magnitude = self.noise_magnitude_base * (1+self._random_noise(0.25, (self.N, self.absolute_state_dim), device=self.dvc))

    def _map_obstacle_init(self):
        '''Init the bound points of the map obstacles
        even_obs_P               丨  (O*P,2)      丨  pygame转换得来
              ↓↓↓
        [并行N份, 然后reshape]
              ↓↓↓
        vec_map_obs_P_shaped   丨  (N,O*P,2,1)  丨  用于编码    丨  用于pygame渲染
              ↓↓↓
        [每次初始化时，编码 (x*window_size+y) ]
              ↓↓↓
        vec_map_bound_code    丨  (N,1,O*P)    丨  雷达扫描
        '''
        map_pyg = pygame.image.load(self.MapObs) # 不能用plt.imread读, 有bug
        map_np = pygame.surfarray.array3d(map_pyg)[:, :, 0]
        self.map_filled_mask = map_np == 0  # 障碍栅格真值掩码(整块填充, 非轮廓), 供 _build_occ_image 栅格化
        x_, y_ = np.where(map_np == 0)  # 障碍物栅格的x,y坐标
        '''注意: 静态障碍物无需对P补齐'''
        even_obs_P = torch.tensor(np.stack((x_, y_), axis=1))  # 障碍物栅格点, (O*P, 2), on cpu
        '''地图障碍物并行N份；然后重塑维度，方便后续扫描'''
        self.vec_map_obs_P_shaped = even_obs_P[None,:,:,None].repeat(self.N,1,1,1).to(self.dvc) # (N,O*P,2,1), on dvc
        # 第ri个env中，所有地图障碍物的x坐标=self.vec_map_obs_P_shaped[self.ri,:,0,0] ; y坐标=self.vec_map_obs_P_shaped[self.ri,:,1,0]

        '''对地图障碍物的x,y坐标进行编码，方便后续扫描'''
        self.vec_map_bound_code = (self.vec_map_obs_P_shaped[:,:,0,0]*self.window_size +self.vec_map_obs_P_shaped[:,:,1,0]).unsqueeze(1) # (N,1,O*P)

    def _static_obstacle_init(self):
        '''Init the bound points of the static obstacles
        even_obs_P               丨  (O*P,2)      丨  pygame绘制得来
              ↓↓↓
        [并行N份, 然后reshape]
              ↓↓↓
        vec_static_obs_P_shaped  丨  (N,O*P,2,1)  丨  用于编码    丨  用于pygame渲染
              ↓↓↓
        [每次初始化时，编码 (x*window_size+y) ]
              ↓↓↓
        vec_static_bound_code    丨  (N,1,O*P)    丨  雷达扫描
        '''
        self.Static_obs_canvas.fill((0, 0, 0))

        '''绘制地图边界'''
        pygame.draw.line(self.Static_obs_canvas, (1, 1, 1), (0, 0),(0, self.window_size), width=self.w-1)
        pygame.draw.line(self.Static_obs_canvas, (1, 1, 1), (0, self.window_size),(self.window_size, self.window_size), width=self.w)
        pygame.draw.line(self.Static_obs_canvas, (1, 1, 1), (self.window_size, self.window_size),(self.window_size, 0), width=self.w+1)
        pygame.draw.line(self.Static_obs_canvas, (1, 1, 1), (self.window_size, 0),(0, 0), width=self.w-1)

        '''对Env0绘制4*sn个障碍:'''
        cdnts = np.random.rand(2, self.sn, self.area**2, 2)*self.d_rect + self.rect_offsets # (start/end,staticObs_numbers,4_rect,x/y)
        for i in range(self.sn):
            for j in range(self.area**2):
                if np.random.rand() < self.generate_rate: # 以概率在每个区域生成障碍物

                    pygame.draw.line(self.Static_obs_canvas, (1, 1, 1), cdnts[0,i,j], cdnts[1,i,j], width=2*self.Obs_R)

        obs_np = pygame.surfarray.array3d(self.Static_obs_canvas)[:, :, 0]
        # 完整障碍位图（含边界线）持久化，供全局规划器构建 c-space 使用；
        # 之后 obs_np 被原地改成轮廓线（obs_np -= b_obs_np），所以在这里先保存一份真值。
        self.world_occ_full = obs_np > 0   # (window_size, window_size) bool, 1cm, 每帧 reset 重新生成
        b_obs_np = ndimage.binary_erosion(obs_np, self.b_kernel).astype(obs_np.dtype)  # 腐蚀障碍物图像
        obs_np -= b_obs_np  # 减去腐蚀图像，提取轮廓线
        self.static_contour_mask = obs_np == 1  # 轮廓真值掩码, 供 _build_occ_image 栅格化(勿用填充位图 world_occ_full)
        x_, y_ = np.where(self.static_contour_mask)  # 障碍物栅格的x,y坐标
        '''注意: 静态障碍物无需对P补齐'''
        even_obs_P = torch.tensor(np.stack((x_, y_), axis=1))  # 障碍物栅格点, (O*P, 2), on cpu
        '''将Env0的障碍物并行N份；然后重塑维度，方便后续扫描'''
        self.vec_static_obs_P_shaped = even_obs_P[None,:,:,None].repeat(self.N,1,1,1).to(self.dvc) # (N,O*P,2,1), on dvc
        # 第ri个env中，所有静态障碍物的x坐标=self.vec_static_obs_P_shaped[self.ri,:,0,0] ; y坐标=self.vec_static_obs_P_shaped[self.ri,:,1,0]

        '''对静态障碍物的x,y坐标进行编码，方便后续扫描'''
        self.vec_static_bound_code = (self.vec_static_obs_P_shaped[:,:,0,0]*self.window_size +self.vec_static_obs_P_shaped[:,:,1,0]).unsqueeze(1) # (N,1,O*P)

        # 把 静态轮廓 ∪ 地图填充 栅格化为占用图像，供 _ld_scan_vec 查表命中检测
        self._build_occ_image()

    def _dynamic_obstacle_init(self):
        '''Init the bound points of the dynamic obstacles:
        vec_dynamic_obs_P         丨  (N,O,P,2)    丨  障碍物运动  丨  障碍物反弹
              ↓↓↓
        [reshape -> 数据联动]
              ↓↓↓
        vec_dynamic_obs_P_shaped  丨  (N,O*P,2,1)  丨  用于编码    丨  用于pygame渲染
              ↓↓↓
        [每次obs移动后，编码 (x*window_size+y) ]
              ↓↓↓
        vec_dynamic_bound_code    丨  (N,1,O*P)    丨  雷达扫描
        '''

        if not self.dynamic_obs:
            self.O = 0  # 关闭动态障碍物：环境只保留静态障碍物

        '''变量初始化'''
        self.Obs_V_tensor = (self._random_noise(self.Obs_V, (self.N, self.O, 1, 2), self.dvc) * self.Obs_refresh_interval *\
                      self.ctrl_interval.reshape(self.N,1,1,1)).to(self.dvc).round().long() # 障碍物的速度, (N,O,1,2)
        self.dynamic_obs_mask = torch.ones((self.N, self.O, 1, 1), dtype=torch.bool, device=self.dvc) # (N,O,1,1), 为False的动态障碍物不能移动
        if self.RdON:
            for i in range(self.N): self.dynamic_obs_mask[i, int(self.O - i * self.O / self.N):, :, :] = False
            # print(self.dynamic_obs_mask.squeeze())

        '''对Env0依次绘制O个障碍:'''
        uneven_obs_P_list = []  # 未补齐的障碍物栅格点坐标
        P_np = np.zeros(self.O, dtype=np.int64) # 记录每个障碍物有多少个Point, 用于后续补齐
        for _ in range(self.O):
            self.Dynamic_obs_canvas.fill((0, 0, 0))
            if self.RdOT and np.random.rand() < 0.5: # 不规则块状障碍物
                thi = np.random.randint(low=20, high=40)
                end_pose = np.random.randint(low=(self.Obs_R+self.w), high=(2*(self.Obs_R+self.w)), size=(2,))
                pygame.draw.line(self.Dynamic_obs_canvas, (1, 1, 1), (0, 0),end_pose, width=thi)
            else: # 环形障碍物
                if self.RdOR: outer_R = 10 + (self.Obs_R-10)*np.random.rand() # 障碍物最小半径10，最大半径Obs_R
                else: outer_R = self.Obs_R
                pygame.draw.circle(self.Dynamic_obs_canvas, (1, 1, 1), (self.Obs_R + self.w, self.Obs_R + self.w), outer_R)

            obs_np = pygame.surfarray.array3d(self.Dynamic_obs_canvas)[:, :, 0]
            b_obs_np = ndimage.binary_erosion(obs_np, self.b_kernel).astype(obs_np.dtype) #腐蚀障碍物图像
            obs_np -=  b_obs_np #减去腐蚀图像，提取轮廓线
            if np.random.rand() < 0.5: obs_np = np.flip(obs_np, (0,)) #水平翻转障碍物
            x_, y_ = np.where(obs_np == 1) # 障碍物栅格的x,y坐标
            bound_gd = torch.tensor(np.stack((x_,y_), axis=1)) # 障碍物栅格点, (unevenP, 2), on cpu
            uneven_obs_P_list.append(bound_gd)
            P_np[_] = bound_gd.shape[0]

        self.P = int(P_np.max()) if self.O > 0 else 1 # 障碍物最大Point数量 (O=0时保留1维,便于广播)
        cP_np = self.P - P_np # 各个障碍物需要补的长度
        '''将各个障碍物栅格点bound_gd统一补齐至P个点，方便存储、运算'''
        even_obs_P = torch.zeros((self.O, self.P, 2), dtype=torch.long) # (O,P,2)
        for _ in range(self.O):
            conpensate = torch.ones(size=(cP_np[_],2), dtype=torch.long)*uneven_obs_P_list[_][0] # on cpu
            even_obs_P[_] = torch.cat((uneven_obs_P_list[_], conpensate))

        '''将Env0的障碍物并行N份, 并随机分散'''
        self.vec_dynamic_obs_P = even_obs_P[None,:,:,:].repeat(self.N,1,1,1).to(self.dvc) # (N,O,P,2), on cpu
        self.vec_dynamic_obs_P += torch.ones((self.N, self.O, 1, 2),dtype=torch.long,device=self.dvc)*(self.R_map-self.Obs_R-self.w)*self.dynamic_obs_mask  # 平移至中心分散
        disperse = self.R_map - 2*(self.Obs_R + self.w)
        self.vec_dynamic_obs_P += torch.randint(-disperse, disperse, (self.N, self.O, 1, 2), device=self.dvc)*self.dynamic_obs_mask # 随机分散
        self.vec_dynamic_obs_P_shaped = self.vec_dynamic_obs_P.reshape(self.N, self.O*self.P, 2, 1) # (N,O,P,2) -> (N,O*P,2,1); on cpu; 与vec_obs_P数据联动
        # 第ri个env中，所有动态障碍物的x坐标=self.vec_dynamic_obs_P_shaped[self.ri,:,0,0] ; y坐标=self.vec_dynamic_obs_P_shaped[self.ri,:,1,0]

        # 随机每个env的最大速度
        if self.ScOV:
            random_idx = torch.randperm(self.N) # 对每个env intance随机，防止env=N的障碍物又最少，速度又最慢
            scale = torch.linspace(1, 0.6, self.N)[random_idx]
            self.max_Obs_Vs = (self.max_Obs_V * scale.reshape((self.N,1,1,1))).long().to(self.dvc)
            # print(self.max_Obs_Vs.squeeze())

    def _rect_in_bound(self, N:int, x:int, y:int, range:int) -> bool:
        '''Check whether the rectangle(center=(x,y), D=2*range) of Envs.N has obstacle.
           All input should be int.'''
        x_min, y_min = max(0, x-range), max(0, y-range)
        x_max, y_max = min(self.window_size, x+range), min(self.window_size, y+range)

        rect = torch.cartesian_prod(torch.arange(x_min, x_max), torch.arange(y_min, y_max)) # (X,2)
        rect_code = (rect[:,0]*self.window_size + rect[:,1]).unsqueeze(-1).to(self.dvc) # (X*2,1)

        return ((rect_code - self.vec_bound_code[N])==0).any() # (X*2,1)-(1,O1*P1+O2*P2+O3*P3)

    def _target_point_init(self, N:int):
        '''Init target point for Envs.N'''
        cnt = 0
        while True:
            cnt += 1
            if cnt > 10000: print("The current map is too crowded to find free space for target init.")
            d, a = self.D * np.random.uniform(0.6,0.9), 6.28 * torch.rand(1)
            x, y = (self.car_state[N, 0].item() + d * torch.cos(a)).int().item(), (self.car_state[N, 1].item() + d * torch.sin(a)).int().item()
            if not ( (self.target_area<x<self.window_size-self.target_area) and (self.target_area<y<self.window_size-self.target_area) ): continue # 不在地图中，重新生成
            if self._rect_in_bound(N, x, y, self.target_area+self.car_radius): continue # 与障碍物重合，重新生成
            self.target_point[N,0], self.target_point[N,1] = x, y
            return x, y

    def _car_loc_init(self, N:int):
        '''Init car location for Envs.N'''
        cnt = 0
        while True:
            cnt += 1
            if cnt > 10000: print("The current map is too crowded to find free space for robot init.")
            loc = torch.randint(low=4*self.car_radius,high=self.window_size-4*self.car_radius,size=(2,),device=self.dvc)
            if self._rect_in_bound(N, loc[0].item(), loc[1].item(), 4*self.car_radius): continue # 与障碍物重合，重新生成
            self.car_state[N,0:2] = loc
            # 朝向不用管，因为target point也会随机生成
            return loc

    def reset(self):
        '''Reset all vectorized Env'''
        #障碍物初始化
        self._static_obstacle_init()
        self._dynamic_obstacle_init()

        # 对动态障碍物进行编码，以用于后续生成小车位置和目标位置时的判断:
        self.vec_bound_code = (self.vec_dynamic_obs_P_shaped[:,:,0,0]*self.window_size +self.vec_dynamic_obs_P_shaped[:,:,1,0]).unsqueeze(1) # (N,1,O*P)
        # 加入静态障碍物：
        self.vec_bound_code = torch.cat((self.vec_static_bound_code,self.vec_bound_code), dim=-1) # (N,1,O1*P1)<->(N,1,O2*P2) => (N,1,O1*P1+O2*P2)
        # 加入地图障碍物
        if self.MapObs: self.vec_bound_code = torch.cat((self.vec_map_bound_code,self.vec_bound_code), dim=-1)

        #小车位置初始化
        self.d2target_pre = torch.zeros(self.N, device=self.dvc)  # Reset() 不产生奖励信号，这里d2target_pre随便赋值即可
        self.car_state.fill_(0)
        for i in range(self.N): self._car_loc_init(i)

        # 目标点初始化：
        for i in range(self.N): self._target_point_init(i)
        self.d2target_now = (self.car_state[:, 0:2] - self.target_point).pow(2).sum(dim=-1).pow(0.5) # (N,), Reset后离目标点的距离,_reward_function和_Normalize会用

        #步数初始化
        self.step_counter_vec.fill_(0)

        #控制指令管道初始化: action5:[0,0]
        self.ctrl_pipe = copy.deepcopy(self.ctrl_pipe_init)

        # 获取初始状态
        observation_vec = self._get_obs() # absolute car state: (N,abs_state_dim)
        # calculate dw, tr, done signals:
        self._reward_function(self.init_pre_action)
        # add noise to unormalized state：
        if self.noise:
            observation_vec += self.noise_magnitude*self._random_noise(1, (self.N,self.absolute_state_dim), self.dvc) # (N,abs_state_dim)

        # Normalize the observation:
        # absolute coordinates will be transformed to relative distance to target
        # absolute orientation will be transformed to relative orientation
        relative_observation_vec = self._Normalize(observation_vec) # (N,abs_state_dim) -> (N,abs_state_dim-1)

        # stack action_state to relative_observation_vec
        act_relative_observation_vec = self._stack_A_to_S(self.init_pre_action, self.init_pre_action, relative_observation_vec) # (N,abs_state_dim-1) -> (N,state_dim)

        # reset()时，偶尔会出现直接done的情况(在Obs_V>self.collide_vec时偶尔会出现)。此时我们可以递归地reset()，直到满足需求。
        if self.done_vec.any():
            self.reset_cnt += 1
            if self.reset_cnt > 100: print(f"Cannot reset env after {self.reset_cnt} tries. May consider reduce Obs_V.")
            return self.reset()

        if self.render_mode == "human": self._render_frame()
        self.reset_cnt = 0

        return act_relative_observation_vec, dict(abs_car_state=self.car_state.clone(), step_cnt = self.step_counter_vec)


    def _AutoReset(self):
        '''Reset done掉的env（没有done的不受影响）'''
        if self.done_vec.any():
            # 1) seset the car pose (only for collided cases)
            CollideEnv_idx = torch.where(self.collide_vec)[0]
            for i in CollideEnv_idx: self._car_loc_init(i)

            # 2) reset the target point
            DoneEnv_idx = torch.where(self.done_vec)[0]
            for i in DoneEnv_idx: self._target_point_init(i)

            # 3) reset the step counter
            self.step_counter_vec[self.done_vec] = 0


    def _obstacle_move(self):
        # 随机障碍物速度(对于每个Env的每一个Obs都随机)
        if self.RdOV: self.Obs_V_tensor += torch.randint(-1,2,(self.N, self.O, 1, 2),device=self.dvc) # 每次速度改变量∈[-1,0,1]

        # 限速
        if self.ScOV: self.Obs_V_tensor.clip_(-self.max_Obs_Vs, self.max_Obs_Vs) # max_Obs_Vs是张量，(N,1,1,1)
        else: self.Obs_V_tensor.clip_(-self.max_Obs_V, self.max_Obs_V) # max_Obs_V是标量

        # 移动障碍物, 注意vec_dynamic_obs_P_shaped会与vec_dynamic_obs_P数据联动
        self.vec_dynamic_obs_P += self.Obs_V_tensor*self.dynamic_obs_mask  # (N,O,P,2) += (N,O,1,2)

        # 对动态障碍物进行编码，以用于后续雷达扫描:
        self.vec_bound_code = (self.vec_dynamic_obs_P_shaped[:,:,0,0]*self.window_size +self.vec_dynamic_obs_P_shaped[:,:,1,0]).unsqueeze(1) # (N,1,O*P)
        # 加入静态障碍物：
        self.vec_bound_code = torch.cat((self.vec_static_bound_code,self.vec_bound_code), dim=-1) # (N,1,O1*P1)<->(N,1,O2*P2) => (N,1,O1*P1+O2*P2)
        # 加入地图障碍物
        if self.MapObs: self.vec_bound_code = torch.cat((self.vec_map_bound_code,self.vec_bound_code), dim=-1)


        # 查看哪些环境的哪些障碍物的x轴/y轴速度需要反向：
        Vx_reverse = ((self.vec_dynamic_obs_P[:, :, :, 0] < self.l_margin) + (self.vec_dynamic_obs_P[:, :, :, 0] > self.h_margin)).any(dim=-1)  # (N,O)
        Vy_reverse = ((self.vec_dynamic_obs_P[:, :, :, 1] < self.l_margin) + (self.vec_dynamic_obs_P[:, :, :, 1] > self.h_margin)).any(dim=-1)  # (N,O)
        # 对越界的障碍物速度反向:
        V_reverse = torch.stack([Vx_reverse, Vy_reverse], dim=2).unsqueeze(2)
        self.Obs_V_tensor[V_reverse] *= -1

    def _ld_scan_vec(self):
        '''Get the scan result (in vectorized worlds) of lidars.
        命中检测改为占用图像查表: 每步 O(N*ld_num) 的 bool gather, 替代原来的
        O(N*ld_num*B) 全障碍物边界集合比对(B=静态轮廓+地图填充+动态, ~38k)。
        静态/地图像素由 _build_occ_image 栅格化为 base_occ_flat(每集 reset 重建,
        集内不变); 动态障碍物按 env 每步移动, 仍用集合比对(训练配置 O=0 时整支关闭)。
        推进/累加/补偿/分组与原实现逐行一致 => 与 verify_ld_scan.py 归档的旧实现 bit-exact。'''
        # 扫描前首先同步雷达与小车位置:
        self.ld_angle = self.ld_angle_interval + self.car_state[:,2,None]# 雷达-小车方向同步, (N, ld_num) + (N, 1) = (N, ld_num)
        self.ld_vectors_wd = torch.stack((torch.cos(self.ld_angle), -torch.sin(self.ld_angle)), dim=2)  # 雷达射线方向, (N,ld_num,2), 注意在unified_cs中是-sin
        self.ld_end_wd = self.car_state[:,None,0:2] + self.car_radius * self.ld_vectors_wd  # 扫描过程中，雷达射线末端世界坐标(初始化于小车轮廓), (N,1,2)+(N,ld_num,2)=(N,ld_num,2)
        self.ld_end_gd = self._world_2_grid(self.ld_end_wd)  # 扫描过程中，雷达射线末端栅格坐标, (N,ld_num,2)
        self.ld_end_code = self.ld_end_gd[:,:,0]*self.window_size + self.ld_end_gd[:,:,1]# 扫描过程中，雷达射线末端栅格坐标的编码值, (N,ld_num)

        # 动态障碍物每env独立、每步移动: 提取其边界编码集合 (N, O*P)。训练配置 O=0 => has_dyn=False, 整支关闭。
        if self.O > 0 and self.P > 0:
            has_dyn = True
            dyn_code = (self.vec_dynamic_obs_P_shaped[:,:,0,0]*self.window_size + self.vec_dynamic_obs_P_shaped[:,:,1,0])  # (N, O*P)
        else:
            has_dyn = False

        # 扫描初始化
        self.ld_scan_result.fill_(0)  # 结果归零, (N, ld_num)
        increment = self.ld_vectors_wd * self.ld_acc  # 每次射出的增量, (N,ld_num,2)

        # 并行式烟花式扫描(PS:当射线穿过地图边界后，会对称地进行扫描。比如穿过上边界，射线会从下边界再射出。这可以模拟地图之外的障碍物。)
        for i in range( int((self.ld_d_range-self.car_radius)/self.ld_acc) + 2 ): # 多扫2次，让最大值超过self.ld_d_range，便于clamp
            # 更新雷达末端位置
            code = self.ld_end_code                          # 当前端点编码, (N,ld_num), dtype与原来一致
            in_win = (code >= 0) & (code < self.w2)          # 越界端点不可能是障碍物集合成员 => 保持"自由/继续扫"
            hit = self.base_occ_flat[code.clamp(0, self.w2 - 1).long()]  # 1D图按2D索引 gather -> (N,ld_num) bool, 静态∪地图命中
            if has_dyn:                                       # 动态障碍物命中(仅当存在), 集合比对保持与旧实现一致
                hit = hit | (code[:, :, None] == dyn_code[:, None, :]).any(dim=-1)
            goon = ~(in_win & hit)                            # 与旧实现的 goon 语义一致: 扫到障碍即停
            self.ld_end_wd += (goon[:,:,None] * increment)  # 更新雷达末端世界坐标,每次射 ld_acc cm #(N, ld_num,1)*(N,ld_num,2)=(N,ld_num,2)
            self.ld_end_gd = self._world_2_grid(self.ld_end_wd)# 更新雷达末端栅格坐标（必须更新，下一轮会调用）, (N,ld_num,2)
            self.ld_end_code = self.ld_end_gd[:, :, 0] * self.window_size + self.ld_end_gd[:, :, 1]# 更新雷达末端栅格坐标编码值, (N,ld_num)
            self.ld_scan_result += (goon * self.ld_acc)# 累计扫描距离 (N, ld_num)

            if (~goon).all(): break # 如果所有ld射线都被挡，则扫描结束

        # 扫描的时候从小车轮廓开始扫的，最后要补偿小车半径的距离; (ld_num, ); torch.tensor
        self.ld_scan_result = (self.ld_scan_result + self.car_radius).clamp(0,self.ld_d_range) #(N, ld_num)

        # 将雷达结果按ld_GN分组，并取每组的最小值作为最终结果
        self.ld_result_grouped, _ = torch.min(self.ld_scan_result.reshape(self.N, self.grouped_ld_num, self.ld_GN), dim=-1, keepdim=False)

    def _reward_function(self, current_a):
        '''Calculate vectorized reward, terminated(dw), truncated(tr), done(dw+tr) signale'''
        self.tr_vec = (self.step_counter_vec > self.max_ep_steps)# truncated signal (N,)
        self.exceed_vec = self.d2target_now > self.D # (N,)
        self.win_vec = self.d2target_now < self.target_area # (N,)
        self.collide_vec = (self.ld_result_grouped < self.collision_trsd).any(dim=-1)  # (N,)
        self.dw_vec = self.exceed_vec + self.win_vec #+ self.collide_vec # terminated signal (N,)
        self.done_vec = self.tr_vec + self.dw_vec # (N,), used for AutoReset

        xy_in_target = self.car_state[:, 0:2] - self.target_point # 小车在以target为原点的坐标系下的坐标, (N,2), 注意这里是无噪声的
        beta = torch.arctan(xy_in_target[:,0] / xy_in_target[:,1]) + torch.pi / 2 + (xy_in_target[:,1] < 0) * torch.pi # (N,)
        alpha = (beta - self.car_state[:, 2]) / torch.pi # (N,)
        alpha += (2 * (alpha < -1) - 2 * (alpha > 1))  # 修复1/2象限、3/4象限, (N,)

        R_distance = ((self.d2target_pre - self.d2target_now)/(self.v_linear_max*self.ctrl_interval.view(-1))).clamp_(-1,1) # 朝目标点移动时得分，背离时扣分。 (N,)∈[-1,1]
        R_orientation = (0.25-alpha.abs().clamp(0,0.25))/0.25  # (-0.25~0~0.25) -> (0,1,0), 朝着目标点时奖励最大,  (N,)
        if self.action_type == 'Discrete':
            R_forward = (current_a==2) # 鼓励使用前进动作(提升移动速度、防止原地滞留) (N,) = 0 or 1
            R_stall = torch.zeros(self.N, device=self.dvc)   # 离散分支保持原奖励不变
            R_angular = torch.zeros(self.N, device=self.dvc) # 离散动作不按角速度惩罚
        else:
            R_forward = torch.zeros(self.N, device=self.dvc) # 连续分支不再按"向前指令"发奖励
            if self.stall_threshold > 0:
                # 实际线速度(经 K 低通滤波后, cm/s)归一化绝对值 ∈[0,1]
                vhat = (self.car_state[:, 3] / self.v_linear_max).abs()
                # |v| < stall_threshold 时给惩罚: 静止罚满(stall_penalty), 越接近阈值罚得越少,
                # >= 阈值不罚。原地逗留与"前后抖动"(高频正负指令在 K 滤波下 |v| 起不来)都被收口;
                # 匀速前进/倒车 |v| 足够大 -> R_stall=0。倒车方向正误由 R_distance 判断。
                R_stall = (self.stall_penalty
                           * (self.stall_threshold - vhat).clamp(min=0.0)
                           / self.stall_threshold)
            else:
                R_stall = torch.zeros(self.N, device=self.dvc)   # threshold=0 -> 关闭该项
            R_angular = current_a[:,1].abs() # |角速度指令| ∈ [0,1]，惩罚原地转圈
        self.reward_vec = (0.5 * R_distance + R_forward
                           - R_stall
                           - self.ang_vel_penalty * R_angular
                           - 0.5)  # -0.5 常数, 防止agent太猥琐到处逗留
        # 注: 移除 R_forward(+1 上限) 与 reverse/Δa 惩罚后, 连续分支非终态稠密奖励上限约
        # 0.5*R_distance - 0.5 (<=0), 正收益主要来自 win(+200)。想要更正向的引导, 可调低
        # 上面的 -0.5 常数或提高 R_distance 权重 0.5。
        # 旧归一化式(参考, 未启用): (0.5*R_distance + R_orientation*R_forward - 0.5*R_retreat_slowdown - 0.25)/1.25

        self.reward_vec[self.win_vec] = self.AWARD
        self.reward_vec[self.exceed_vec] = self.PUNISH
        self.reward_vec[self.collide_vec] = self.PUNISH

    def _Normalize(self, observation) -> torch.tensor:
        '''Normalize the raw observations (N,abs_state_dim) to relative observations (N,abs_state_dim-1)'''
        # 1) Normalize the orientation (only for pygame coordinates):
        xy_in_target = observation[:,0:2] - self.target_point # 小车在以target为原点的坐标系下的坐标, (N,2), 注意这里可能带噪声的
        beta = torch.arctan(xy_in_target[:,0] / xy_in_target[:,1]) + torch.pi / 2 + (xy_in_target[:,1] < 0) * torch.pi # (N,)
        observation[:, 2] = (beta - observation[:, 2]) / torch.pi
        observation[:, 2] += ( 2*(observation[:, 2] < -1) - 2*(observation[:, 2] > 1) ) # 修复1/2象限、3/4象限

        # 2) Stack d2target_now with observation[:,2:]
        new_obs = torch.hstack((self.d2target_now.unsqueeze(-1), observation[:,2:])) # (N,abs_state_dim-1), [D2T,alpha,Vlinear,Vangle,ld0,...ldn]

        # 3) Normalize new_obs:
        return new_obs/self.state_upperbound

    def _stack_A_to_S(self, current_a, real_a, observation) -> torch.tensor:
        """
        transform action (N,) to action_state (N,2) and
        stack action_state (N,2) to the observation"""
        if self.action_type == 'Discrete':
            return torch.cat((self.a_state[current_a], self.a_state[real_a], observation), dim=1) # (N,2)+(N,2)->(N,abs_state_dim-1) => (N,state_dim)
        else:
            return torch.cat((current_a, real_a, observation), dim=1) # (N,2)+(N,2)->(N,abs_state_dim-1) => (N,state_dim)


    def _get_obs(self) -> torch.tensor:
        '''Return: Un-normalized and un-noised observation [dx, dy, theta, v_linear, v_angular, lidar_results(0), ..., lidar_results(n-1)] in shape (N,abs_state_dim) '''
        # 1.障碍物运动：
        self.Obs_refresh_counter += 1
        if self.Obs_refresh_counter > self.Obs_refresh_interval:
            self._obstacle_move()
            self.Obs_refresh_counter = 1

        # 2.雷达扫描：
        self._ld_scan_vec()

        # 3.制作observation
        observation_vec = torch.concat((self.car_state, self.ld_result_grouped), dim=-1) #(N, 5) cat (N, grouped_ld_num) = (N, 23)
        return observation_vec

    def _Discrete_Kinematic_model_vec(self, a) -> torch.tensor:
        ''' V_now = K*V_previous + (1-K)*V_target
            Input: discrete action index, (N,)
            Output: [v_l, v_l, v_a], (N,3)'''
        self.car_state[:,3:5] = self.K * self.car_state[:,3:5] + (1-self.K)*self.a_space[self.arange_constant,a] # self.a_space[a] is (N,2)
        return torch.stack((self.car_state[:,3],self.car_state[:,3],self.car_state[:,4]),dim=1) # [v_l, v_l, v_a], (N,3)


    def _Continuous_Kinematic_model_vec(self, a) -> torch.tensor:
        ''' V_now = K*V_previous + (1-K)*V_target
            Input: continuous action, (N,2)
            Output: [v_l, v_l, v_a], (N,3)'''
        self.car_state[:,3:5] = self.K * self.car_state[:,3:5] + (1-self.K)*self.continuous_scale*a # a.shape = (N,2)
        return torch.stack((self.car_state[:,3],self.car_state[:,3],self.car_state[:,4]),dim=1) # [v_l, v_l, v_a], (N,3)

    def step(self,current_a):
        """
        When self.action_type=='Discrete', 'current_a' should be a vectorized discrete action of dim (N,) on self.dvc
        For self.action_type=='Continuous', 'current_a' should be a vectorized continuous action of dim (N,2) on self.dvc
        """

        '''Domain randomization'''
        self.step_counter_vec += 1
        # domain randomization in a fixed frequency
        self.step_counter_DR += self.N
        if self.DR and (self.step_counter_DR > self.DR_freq):
            self.step_counter_DR = 0
            self._Domain_Randomization()

        '''Update car state: [dx, dy, theta, v_linear, v_angular]'''
        # control delay mechanism
        self.ctrl_pipe.append(current_a) # current_a is the action mapped by the current state
        real_a = self.ctrl_pipe.popleft() # real_a is the delayed action,

        # calculate and update the velocity of the car based on the delayed action and the Kinematic_model
        if self.action_type == "Discrete": velocity = self._Discrete_Kinematic_model_vec(real_a) # [v_l, v_l, v_a], (N,3)
        else: velocity = self._Continuous_Kinematic_model_vec(real_a) # [v_l, v_l, v_a], (N,3)

        # calculate and update the [dx,dy,orientation] of the car
        self.d2target_pre = (self.car_state[:, 0:2] - self.target_point).pow(2).sum(dim=-1).pow(0.5) # (N,), 执行动作前离目标点的距离
        self.car_state[:,0:3] += self.ctrl_interval * velocity * torch.stack((torch.cos(self.car_state[:,2]),
                                                                              -torch.sin(self.car_state[:,2]),
                                                                              torch.ones(self.N,device=self.dvc)), dim=1)
        self.d2target_now = (self.car_state[:, 0:2] - self.target_point).pow(2).sum(dim=-1).pow(0.5) # (N,), 执行动作后离目标点的距离,_reward_function和_Normalize会用

        # keep the orientation between [0,2pi]
        self.car_state[:,2] %= (2 * torch.pi)

        '''Update observation: observation -> add_noise -> normalize -> stack[cA,rA,O]'''
        # get next obervation
        observation_vec  = self._get_obs()

        # calculate reward, dw, tr, done signals
        self._reward_function(current_a)

        # add noise to unormalized state：
        if self.noise:
            observation_vec += self.noise_magnitude*self._random_noise(1, (self.N,self.absolute_state_dim), self.dvc) # (N, 23)

        # Normalize the observation:
        # absolute coordinates will be transformed to relative distance to target
        # absolute orientation will be transformed to relative orientation
        relative_observation_vec = self._Normalize(observation_vec) # (N,22)

        # stack action_state to relative_observation_vec
        act_relative_observation_vec = self._stack_A_to_S(current_a, real_a, relative_observation_vec) # (N,22) -> (N,26)

        '''Render and AutoReset'''
        # render the current frame
        if self.render_mode == "human": self._render_frame()

        # reset some of the envs based on the done_vec signal
        self._AutoReset()

        return act_relative_observation_vec, \
            self.reward_vec.clone(), \
            self.dw_vec.clone(), \
            self.tr_vec.clone(), \
            dict(abs_car_state=self.car_state.clone(), step_cnt = self.step_counter_vec)

    def occupied_grid_map(self) -> np.ndarray:
        """Get the occupied grid map (render_mode must be "human")
        The ogm can be rendered via 'plt.imshow(self.ogm)' """
        return self.ogm # (window_size, window_size, 3)

    def _draw_robot(self):
        """Draw the robot body + heading line onto `self.canvas`.

        Overridable so subclasses (e.g. `SparrowCspaceEnv`) can render the
        observation footprint polygon instead of Sparrow's collision circle.
        """
        # draw robot on canvas
        pygame.draw.circle(
            self.canvas,
            (200, 128, 250),
            self._world_2_grid(self.car_state[self.ri,0:2]).cpu().numpy(),
            self.car_radius,
        )
        # draw robot orientation on canvas
        head = self.car_state[self.ri,0:2].cpu() + self.car_radius * torch.tensor([torch.cos(self.car_state[self.ri,2]), -torch.sin(self.car_state[self.ri,2])])
        pygame.draw.line(
            self.canvas,
            (0, 255, 255),
            self._world_2_grid(self.car_state[self.ri,0:2]).cpu().numpy(),
            self._world_2_grid(head).numpy(),
            width=2
        )

    def _draw_lidar_rays(self):
        """Draw the lidar scan rays onto `self.canvas`.

        Overridable: `SparrowCspaceEnv` replaces this with the NavRL c-space cost
        rays (farthest reach over all theta layers) instead of the raw lidar scan.
        """
        ld_result = self.ld_scan_result[self.ri].cpu().clone()  # (ld_num,), on cpu
        ld_real_sta_gd = self._world_2_grid(self.car_state[self.ri, 0:2]).cpu().numpy()  # (2,)
        ld_real_end_gd = self._world_2_grid(
            self.car_state[self.ri, 0:2].cpu()
            + ld_result.unsqueeze(-1) * self.ld_vectors_wd[self.ri].cpu()
        ).numpy()  # (ld_num, 2)
        for i in range(self.ld_num):
            e = 255 * ld_result[i] / self.ld_d_range
            pygame.draw.aaline(self.canvas, (255 - e, 0, e),
                               ld_real_sta_gd, ld_real_end_gd[i])

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size , self.window_size ))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # init canvas
        if self.canvas is None :
            self.canvas = pygame.Surface((self.window_size , self.window_size ))

        # draw obstacles:
        self.obs_canvas_torch.fill_(255)
        if self.MapObs: self.obs_canvas_torch[self.vec_map_obs_P_shaped[self.ri,:,0],self.vec_map_obs_P_shaped[self.ri,:,1]] = 0
        self.obs_canvas_torch[self.vec_static_obs_P_shaped[self.ri,:,0],self.vec_static_obs_P_shaped[self.ri,:,1]] = 0
        self.obs_canvas_torch[self.vec_dynamic_obs_P_shaped[self.ri,:,0],self.vec_dynamic_obs_P_shaped[self.ri,:,1]] = 105 #101, 104, 105
        obstacles = pygame.surfarray.make_surface(self.obs_canvas_torch.numpy())
        self.canvas.blit(obstacles, self.canvas.get_rect())
        self.ogm = np.transpose(np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2))  # occupied grid maps

        # draw auxiliary information
        if self.draw_auxiliary:
            # 静态障碍物生成区域:
            for i in range(self.area**2):
                pygame.draw.rect(self.canvas, (128, 128, 128), (self.rect_offsets[i,0], self.rect_offsets[i,1], self.d_rect, self.d_rect),width=2)


        # draw target area
        pygame.draw.circle(self.canvas, (100, 225, 100), self.target_point[self.ri].cpu().numpy(), self.target_area, 2)


        # draw sensor rays (lidar by default; c-space cost rays for SparrowCspaceEnv)
        self._draw_lidar_rays()

        # draw collision threshold on canvas
        pygame.draw.circle(
            self.canvas,
            (64,64,64),
            self._world_2_grid(self.car_state[self.ri,0:2]).cpu().numpy(),
            self.collision_trsd,
        )

        # draw robot body + orientation (circle by default; footprint polygon for c-space envs)
        self._draw_robot()

        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(self.canvas, self.canvas.get_rect())
        pygame.event.pump()
        if not getattr(self, "manual_flip", False):
            pygame.display.update()

        if self.render_speed == 'real':
            self.clock.tick(int(1 / self.render_rate))
        elif self.render_speed == 'fast':
            self.clock.tick(0)
        elif self.render_speed == 'slow':
            self.clock.tick(30)
        else:
            print('Wrong Render Speed, only "real"; "fast"; "slow" is acceptable.')


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

"""--------------------------------------------------------------------------------------------------------"""

def str2bool(v):
    '''Fix the bool BUG for argparse: transfer string to bool'''
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1', 'T'): return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0', 'F'): return False
    else: print('Wrong Input Type!')

"""--------------------------------------------------------------------------------------------------------"""

class Sparrow_PlayGround(Sparrow):
    def __init__(self, **params):
        super().__init__(**params)
        self.waypoints = deque()
        self.global_path = deque()
        self.N = 1
        self.ri = 0

    def reset(self):
        '''Reset Env.ri'''
        #障碍物初始化
        self._static_obstacle_init()
        self._dynamic_obstacle_init()

        # 对动态障碍物进行编码，以用于后续生成小车位置和目标位置时的判断:
        self.vec_bound_code = (self.vec_dynamic_obs_P_shaped[:,:,0,0]*self.window_size +self.vec_dynamic_obs_P_shaped[:,:,1,0]).unsqueeze(1) # (N,1,O*P)
        # 加入静态障碍物：
        self.vec_bound_code = torch.cat((self.vec_static_bound_code,self.vec_bound_code), dim=-1) # (N,1,O1*P1)<->(N,1,O2*P2) => (N,1,O1*P1+O2*P2)
        # 加入地图障碍物
        if self.MapObs: self.vec_bound_code = torch.cat((self.vec_map_bound_code,self.vec_bound_code), dim=-1)

        #小车位置初始化
        self.d2target_pre = torch.zeros(self.N, device=self.dvc)  # Reset() 不产生奖励信号，这里pre随便赋值即可
        self.car_state.fill_(0)
        init_xy = self._car_loc_init(self.ri)

        # 重置global_path, waypoints, target_point
        self.global_path.clear()
        self.waypoints.clear()
        self.global_path.append(init_xy.cpu().numpy())
        self.waypoints.append(init_xy.cpu().numpy())
        self.target_point = init_xy

        self.d2target_now = (self.car_state[:, 0:2] - self.target_point).pow(2).sum(dim=-1).pow(0.5) # (N,), Reset后离目标点的距离,_reward_function和_Normalize会用

        #步数初始化
        self.step_counter_vec.fill_(0)

        #控制指令管道初始化: action5:[0,0]
        self.ctrl_pipe = copy.deepcopy(self.ctrl_pipe_init)

        # 获取初始状态
        observation_vec = self._get_obs() # absolute car state, (N,23)
        # calculate dw, tr, done signals:
        self._reward_function(self.init_pre_action)
        # add noise to unormalized state：
        if self.noise:
            observation_vec += self.noise_magnitude*self._random_noise(1, (self.N,self.absolute_state_dim), self.dvc) # (N, absolute_state_dim)

        # Normalize the observation:
        # absolute coordinates will be transformed to relative distance to target
        # absolute orientation will be transformed to relative orientation
        relative_observation_vec = self._Normalize(observation_vec) # (N,23) -> (N,22)

        # stack action_state to relative_observation_vec: (N,absolute_state_dim-1) -> (N,state_dim)
        act_relative_observation_vec = self._stack_A_to_S(self.init_pre_action, self.init_pre_action, relative_observation_vec)

        # if self.done_vec.any(): raise RuntimeError('Done Signal in Reset State!')
        if self.render_mode == "human": self._render_frame()

        return act_relative_observation_vec, dict(abs_car_state=self.car_state.clone(), step_cnt = self.step_counter_vec)

    def _reward_function(self, current_a):
        """只计算是否到达，是否碰撞"""
        self.win_vec = self.d2target_now < self.target_area # (N,)
        self.collide_vec = (self.ld_result_grouped < self.collision_trsd).any(dim=-1)  # (N,)
        self.dw_vec = self.win_vec #+ self.collide_vec # terminated signal (N,)

    def step(self,current_a):
        """
        When self.action_type=='Discrete', 'current_a' should be a vectorized discrete action of dim (N,) on self.dvc
        For self.action_type=='Continuous', 'current_a' should be a vectorized continuous action of dim (N,2) on self.dvc
        """

        self.step_counter_vec += 1

        # control delay mechanism
        self.ctrl_pipe.append(current_a) # current_a is the action mapped by the current state
        real_a = self.ctrl_pipe.popleft() # real_a is the delayed action,

        # calculate and update the velocity of the car based on the delayed action and the Kinematic_model
        if self.action_type == "Discrete": velocity = self._Discrete_Kinematic_model_vec(real_a) # [v_l, v_l, v_a], (N,3)
        else: velocity = self._Continuous_Kinematic_model_vec(real_a) # [v_l, v_l, v_a], (N,3)

        # calculate and update the [dx,dy,orientation] of the car
        self.d2target_pre = (self.car_state[:, 0:2] - self.target_point).pow(2).sum(dim=-1).pow(0.5) # (N,), 执行动作前离目标点的距离
        self.car_state[:,0:3] += self.ctrl_interval * velocity * torch.stack((torch.cos(self.car_state[:,2]),
                                                                              -torch.sin(self.car_state[:,2]),
                                                                              torch.ones(self.N,device=self.dvc)), dim=1)
        self.d2target_now = (self.car_state[:, 0:2] - self.target_point).pow(2).sum(dim=-1).pow(0.5) # (N,), 执行动作后离目标点的距离,_reward_function和_Normalize会用

        # keep the orientation between [0,2pi]
        self.car_state[:,2] %= (2 * torch.pi)

        # get next obervation
        observation_vec  = self._get_obs()

        # calculate reward, dw, tr, done signals
        self._reward_function(current_a)

        # change target point
        if self.win_vec and len(self.waypoints) > 1:
            self.waypoints.popleft()
            self.target_point = torch.tensor(self.waypoints[0], device=self.dvc)

        # Normalize the observation:
        # absolute coordinates will be transformed to relative distance to target
        # absolute orientation will be transformed to relative orientation
        relative_observation_vec = self._Normalize(observation_vec) # (N,31)

        # stack action_state to relative_observation_vec
        act_relative_observation_vec = self._stack_A_to_S(current_a, real_a, relative_observation_vec) # (N,31) -> (N,35)

        # render the current frame
        if self.render_mode == "human": self._render_frame()

        # reset some of the envs based on the done_vec signal
        self._AutoReset()

        return act_relative_observation_vec, \
            self.reward_vec.clone(), \
            self.dw_vec.clone(), \
            self.tr_vec.clone(), \
            dict(abs_car_state=self.car_state.clone(), step_cnt = self.step_counter_vec)

    def _AutoReset(self):
        '''Reset collide掉的env（没有done的不受影响）'''
        if self.collide_vec.any():

            # 1) seset the car pose
            init_xy = self._car_loc_init(self.ri)

            # 2) reset waypoints and global path
            self.global_path.clear()
            self.waypoints.clear()
            self.global_path.append(init_xy.cpu().numpy())
            self.waypoints.append(init_xy.cpu().numpy())
            self.target_point = init_xy

            # 3) reset the step counter
            self.step_counter_vec[self.ri] = 0


    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size , self.window_size ))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # init canvas
        if self.canvas is None :
            self.canvas = pygame.Surface((self.window_size , self.window_size ))

        # draw obstacles:
        self.obs_canvas_torch.fill_(255)
        if self.MapObs: self.obs_canvas_torch[self.vec_map_obs_P_shaped[self.ri,:,0],self.vec_map_obs_P_shaped[self.ri,:,1]] = 0
        self.obs_canvas_torch[self.vec_static_obs_P_shaped[self.ri,:,0],self.vec_static_obs_P_shaped[self.ri,:,1]] = 0
        self.obs_canvas_torch[self.vec_dynamic_obs_P_shaped[self.ri,:,0],self.vec_dynamic_obs_P_shaped[self.ri,:,1]] = 105 #101, 104, 105
        obstacles = pygame.surfarray.make_surface(self.obs_canvas_torch.numpy())
        self.canvas.blit(obstacles, self.canvas.get_rect())
        self.ogm = np.transpose(np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2))  # occupied grid maps


        # draw sensor rays (lidar by default; c-space cost rays for SparrowCspaceEnv)
        self._draw_lidar_rays()

        # draw collision threshold on canvas
        pygame.draw.circle(
            self.canvas,
            (64,64,64),
            self._world_2_grid(self.car_state[self.ri,0:2]).cpu().numpy(),
            self.collision_trsd,
        )

        # draw robot body + orientation (circle by default; footprint polygon for c-space envs)
        self._draw_robot()

        # draw global path
        if self.global_path:
            for i in range(len(self.global_path)-1):
                pygame.draw.line(self.canvas, (255,200,15), self.global_path[i],self.global_path[i+1], width=4)

        if self.waypoints:
            # draw target point
            pygame.draw.circle(self.canvas,(100, 225, 100),self.waypoints[0],self.target_area,2)

            # draw waypoints
            for i in range(1, len(self.waypoints)):
                pygame.draw.circle(self.canvas,(255,200,15),self.waypoints[i],self.target_area,2)


        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(self.canvas, self.canvas.get_rect())
        pygame.event.pump()
        if not getattr(self, "manual_flip", False):
            pygame.display.update()

        if self.render_speed == 'real':
            self.clock.tick(int(1 / self.render_rate))
        elif self.render_speed == 'fast':
            self.clock.tick(0)
        elif self.render_speed == 'slow':
            self.clock.tick(30)
        else:
            print('Wrong Render Speed, only "real"; "fast"; "slow" is acceptable.')


        # record waypoints:
        ev = pygame.event.get()
        for event in ev:
            if event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_click(np.array(pygame.mouse.get_pos()))

    def _handle_click(self, xy):
        '''Handle a mouse click in the playground window.
        默认行为：直接把点击点追加为航点（若在最大局部规划阈值内）。
        子类（如 SparrowCspacePlayground）可覆写为「点击 → 全局规划 → 整条路径入队」。'''
        distance = np.linalg.norm(xy - self.waypoints[-1])
        if distance > self.D:
            print("Distance exceeds maximal local planning threshold!")
        else:
            self.global_path.append(xy)
            self.waypoints.append(xy)
