import math
import os
import random
import subprocess
import time
import xml.etree.ElementTree as ET
from os import path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.1


def _parse_sdf_pose(pose_str):
    """Parse SDF pose string 'x y z roll pitch yaw' → (x, y, z, roll, pitch, yaw)."""
    parts = [float(p) for p in pose_str.split()]
    while len(parts) < 6:
        parts.append(0.0)
    return tuple(parts[:6])


def _rotate_2d(x, y, yaw):
    """Rotate point (x, y) by yaw radians in the XY plane."""
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (x * c - y * s, x * s + y * c)


def parse_world_obstacles(world_path, margin=0.3):
    """Parse an SDF world file and extract axis-aligned bounding boxes of static obstacles.

    Walks every <model> → <link> → <collision> chain, resolves the full 6-DOF
    transform into the world frame, and returns a list of *(x_min, x_max, y_min,
    y_max)* tuples.  Models whose name contains ``cardboard_box`` are skipped
    because they are randomized at runtime.

    Args:
        world_path: path to the ``.world`` SDF file.
        margin:     extra padding (metres) added to every AABB.

    Returns:
        list[tuple[float, float, float, float]]  — one AABB per collision box.
    """
    tree = ET.parse(world_path)
    root = tree.getroot()
    obstacles = []

    for model in root.iter('model'):
        name = model.get('name', '')
        if 'cardboard_box' in name:
            continue

        # Model-level pose (world frame)
        model_pose_elem = model.find('pose')
        model_pose = _parse_sdf_pose(
            model_pose_elem.text if model_pose_elem is not None else '0 0 0 0 0 0')
        mx, my, _, _, _, myaw = model_pose

        for link in model.iter('link'):
            link_pose_elem = link.find('pose')
            link_pose = _parse_sdf_pose(
                link_pose_elem.text if link_pose_elem is not None else '0 0 0 0 0 0')
            lx, ly, _, _, _, lyaw = link_pose

            for collision in link.iter('collision'):
                coll_pose_elem = collision.find('pose')
                coll_pose = _parse_sdf_pose(
                    coll_pose_elem.text if coll_pose_elem is not None else '0 0 0 0 0 0')
                cx, cy, _, _, _, cyaw = coll_pose

                geometry = collision.find('geometry')
                if geometry is None:
                    continue

                # ---- Determine half-extents in collision-local XY frame ----
                half_x, half_y = None, None  # will be set by the geometry handler

                box = geometry.find('box')
                cylinder = geometry.find('cylinder')
                mesh = geometry.find('mesh')

                if box is not None:
                    size_elem = box.find('size')
                    if size_elem is None:
                        continue
                    sx, sy, _sz = [float(v) for v in size_elem.text.split()]
                    half_x, half_y = sx / 2.0, sy / 2.0
                elif cylinder is not None:
                    radius_elem = cylinder.find('radius')
                    if radius_elem is None:
                        continue
                    r = float(radius_elem.text)
                    half_x, half_y = r, r
                elif mesh is not None:
                    # Mesh geometry cannot be reliably reduced to an AABB;
                    # skip it (rare — typically only small decorative models).
                    continue
                else:
                    continue  # sphere, plane, etc. — ignore

                if half_x is None:
                    continue

                # Compose transforms: model → link → collision → box corners
                total_yaw = myaw + lyaw + cyaw

                # World position of collision origin
                #   p_world = R(myaw) * (R(lyaw) * [cx,cy] + [lx,ly]) + [mx,my]
                cx_rot, cy_rot = _rotate_2d(cx, cy, lyaw)
                wx = mx + _rotate_2d(lx + cx_rot, ly + cy_rot, myaw)[0]
                wy = my + _rotate_2d(lx + cx_rot, ly + cy_rot, myaw)[1]

                # Corners in collision-local frame, rotated to world
                corners = [
                    (-half_x, -half_y), (half_x, -half_y),
                    (-half_x, half_y),  (half_x, half_y),
                ]
                world_corners = [_rotate_2d(cx2, cy2, total_yaw) for cx2, cy2 in corners]
                xs = [wx + wcx for wcx, _ in world_corners]
                ys = [wy + wcy for _, wcy in world_corners]

                obstacles.append((
                    min(xs) - margin, max(xs) + margin,
                    min(ys) - margin, max(ys) + margin,
                ))

    return obstacles


def _resolve_world_path(launchfile, port="11311"):
    """Try to find the absolute path of the .world file referenced by a roslaunch file.

    Walks the launch-file include chain and returns the first ``.world`` path found,
    or **None** if the resolution fails.
    """
    if launchfile.startswith("/"):
        fullpath = launchfile
    else:
        fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
    if not path.exists(fullpath):
        return None

    try:
        tree = ET.parse(fullpath)
        root = tree.getroot()
    except ET.ParseError:
        return None

    # Collect include tags
    include_files = []
    for include in root.iter('include'):
        file_attr = include.get('file')
        if file_attr:
            include_files.append(file_attr)

    # Try to resolve each include relative to ROS package paths
    ros_package_path = os.environ.get("ROS_PACKAGE_PATH", "")
    search_dirs = [os.path.dirname(fullpath)]
    for p in ros_package_path.split(":"):
        p = p.strip()
        if p:
            search_dirs.append(p)

    for inc in include_files:
        # inc looks like "$(find multi_robot_scenario)/launch/empty_world.launch"
        # Extract the package-relative path
        if inc.startswith("$(find ") and ")" in inc:
            rest = inc[inc.index(")") + 1:]
            pkg_end = inc.index(")")
            pkg_name = inc[7:pkg_end].strip()
            for d in search_dirs:
                pkg_dir = os.path.join(d, pkg_name) if not d.endswith(pkg_name) else d
                candidate = os.path.join(pkg_dir, rest.lstrip("/"))
                if path.exists(candidate):
                    # Now parse this launch file for the world_name arg
                    world_path = _find_world_in_launch(candidate, search_dirs, pkg_name, d)
                    if world_path:
                        return world_path
        elif inc.endswith(".world"):
            for d in search_dirs:
                candidate = os.path.join(d, inc)
                if path.exists(candidate):
                    return candidate
        else:
            # Simple relative path
            candidate = os.path.join(os.path.dirname(fullpath), inc)
            if path.exists(candidate):
                world_path = _find_world_in_launch(candidate, search_dirs, "", "")
                if world_path:
                    return world_path

    return None


def _find_world_in_launch(launch_path, search_dirs, pkg_name, pkg_dir):
    """Parse a launch file, find the ``world_name`` arg and resolve it to a path."""
    try:
        tree = ET.parse(launch_path)
        root = tree.getroot()
    except ET.ParseError:
        return None

    # Find default world_name arg value
    default_world = None
    world_name_from_arg = None
    for arg in root.iter('arg'):
        if arg.get('name') == 'world_name':
            default_world = arg.get('default')
            world_name_from_arg = arg.get('value')
    world_name = world_name_from_arg or default_world

    if world_name is None:
        # Check if the launch file directly passes a world file name to gzserver
        for node in root.iter('node'):
            args = node.get('args', '')
            if '.world' in args:
                # Extract the world file name from args
                parts = args.split()
                for i, p in enumerate(parts):
                    if p.endswith('.world'):
                        world_name = p
                        break

    if world_name is None:
        return None

    # Resolve world_name to a full path
    launch_dir = os.path.dirname(launch_path)
    for d in [launch_dir, pkg_dir] + search_dirs:
        d = d.strip()
        if not d:
            continue
        candidate = os.path.join(d, world_name)
        if path.exists(candidate):
            return candidate
        # Try as relative to launch dir or pkg dir
        if pkg_name:
            candidate = os.path.join(d, pkg_name, "launch", world_name)
            if path.exists(candidate):
                return candidate

    return None


# Backward-compatible check_pos: used when no world info is available.
# Kept as a module-level function for external callers.
def check_pos(x, y, obstacles=None, boundary=4.5):
    """Return True if *(x, y)* is free of obstacles and within the world boundary.

    Args:
        x, y:       position to check.
        obstacles:  list of ``(x_min, x_max, y_min, y_max)`` AABBs, or **None**
                    to use the legacy hard-coded rectangles for TD3.world.
        boundary:   absolute x/y limit.  Points outside ``[-boundary, boundary]``
                    are rejected.
    """
    # Boundary check
    if abs(x) > boundary or abs(y) > boundary:
        return False

    if obstacles is None:
        # Legacy hard-coded obstacles for TD3.world (backward compatibility)
        if -3.8 > x > -6.2 and 6.2 > y > 3.8:
            return False
        if -1.3 > x > -2.7 and 4.7 > y > -0.2:
            return False
        if -0.3 > x > -4.2 and 2.7 > y > 1.3:
            return False
        if -0.8 > x > -4.2 and -2.3 > y > -4.2:
            return False
        if -1.3 > x > -3.7 and -0.8 > y > -2.7:
            return False
        if 4.2 > x > 0.8 and -1.8 > y > -3.2:
            return False
        if 4 > x > 2.5 and 0.7 > y > -3.2:
            return False
        if 6.2 > x > 3.8 and -3.3 > y > -4.2:
            return False
        if 4.2 > x > 1.3 and 3.7 > y > 1.5:
            return False
        if -3.0 > x > -7.2 and 0.5 > y > -1.5:
            return False
        return True

    # Dynamic check against parsed obstacle AABBs
    for (x_min, x_max, y_min, y_max) in obstacles:
        if x_min < x < x_max and y_min < y < y_max:
            return False
    return True


class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim, use_cnn=False,
                 ray_h_beams=36, ray_v_layers=21, max_ray_range=5.0,
                 safety_reward_weight=1.0, speed_reward_weight=1.0,
                 port="11311", world_name=None):
        self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0

        self.goal_x = 1
        self.goal_y = 0.0

        self.upper = 5.0
        self.lower = -5.0
        self.velodyne_data = np.ones(self.environment_dim) * 10
        self.last_odom = None
        self.last_distance = 0.0  # for progress reward

        # ---- Reward weights ----
        self.safety_reward_weight = safety_reward_weight
        self.speed_reward_weight = speed_reward_weight

        # ---- CNN mode: cost_rays from c_space_mapper_node ----
        self.use_cnn = use_cnn
        self.ray_h_beams = ray_h_beams        # horizontal beams (default 36 = 10° res)
        self.ray_v_layers = ray_v_layers      # theta layers from c-space (default 21)
        self.max_ray_range = max_ray_range    # maximum ray range for preprocessing
        # Buffer: initialized to max_range (no collision detected yet)
        self.cost_rays_data = np.ones((ray_h_beams, ray_v_layers)) * max_ray_range
        self.world_name = world_name

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        self.gaps = [[-np.pi - 0.03, -np.pi + 2 * np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + 2 * np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        subprocess.Popen(["roscore", "-p", port])

        print("Roscore launched!")

        # Launch the simulation with the given launchfile name
        rospy.init_node("gym", anonymous=True)

        # Ensure use_sim_time is set before any nodes start publishing TF
        rospy.set_param("/use_sim_time", True)

        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")

        # ---- Parse world obstacles from the SDF file ----
        self.obstacles = None   # list of (x_min,x_max,y_min,y_max) or None
        self.boundary = 4.5     # default world boundary
        world_path = _resolve_world_path(launchfile, port)
        if world_path is not None and path.exists(world_path):
            try:
                self.obstacles = parse_world_obstacles(world_path, margin=0.3)
                print("Parsed %d obstacle(s) from %s" % (len(self.obstacles), world_path))
                # Infer boundary from the innermost edge of boundary walls.
                # Boundary walls are long (> 8 m) and thin (< 1.0 m after margin).
                for (x_min, x_max, y_min, y_max) in self.obstacles:
                    span_x = x_max - x_min
                    span_y = y_max - y_min
                    if span_x > 8 and span_y < 1.0:
                        # North/South wall: use the inner y edge as boundary
                        if abs(y_max) < abs(y_min):
                            self.boundary = max(self.boundary, abs(y_max))
                        else:
                            self.boundary = max(self.boundary, abs(y_min))
                    if span_y > 8 and span_x < 1.0:
                        # East/West wall: use the inner x edge as boundary
                        if abs(x_max) < abs(x_min):
                            self.boundary = max(self.boundary, abs(x_max))
                        else:
                            self.boundary = max(self.boundary, abs(x_min))
                # Add a small inward offset so the robot doesn't spawn right on the wall
                self.boundary -= 0.5
            except Exception as e:
                print("WARNING: Failed to parse world obstacles: %s" % e)
                self.obstacles = None
        else:
            print("WARNING: Could not resolve world path — "
                  "falling back to legacy obstacle checks")

        # ---- CNN mode: launch c_space_mapper pipeline ----
        if self.use_cnn:
            project_root = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
            cspace_src = os.path.join(project_root, "catkin_ws", "src")

            # Ensure ROS can find the c_space_mapper packages
            ros_env = os.environ.copy()
            if "ROS_PACKAGE_PATH" in ros_env:
                if cspace_src not in ros_env["ROS_PACKAGE_PATH"]:
                    ros_env["ROS_PACKAGE_PATH"] = (
                        cspace_src + ":" + ros_env["ROS_PACKAGE_PATH"])
            else:
                ros_env["ROS_PACKAGE_PATH"] = cspace_src

            cspace_launch = os.path.join(
                cspace_src, "c_space_mapper", "launch", "c_space_mapper.launch")
            if path.exists(cspace_launch):
                subprocess.Popen(
                    ["roslaunch", "-p", port, cspace_launch], env=ros_env)
                print("c_space_mapper launched!")
                # Wait for pipeline: lidar_to_occupancy → c_space_mapper
                time.sleep(3)
            else:
                print("WARNING: c_space_mapper launch not found at %s"
                      % cspace_launch)

        # Set up the ROS publishers and subscribers
        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        self.velodyne = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        )
        self.odom = rospy.Subscriber(
            "/r1/odom", Odometry, self.odom_callback, queue_size=1
        )
        if self.use_cnn:
            self.cost_rays_sub = rospy.Subscriber(
                "/cost_rays", Float32MultiArray,
                self.cost_rays_callback, queue_size=1
            )

    # Read velodyne pointcloud and turn it into distance data, then select the
    # minimum value for each angle range as state representation.
    # The raw velodyne_data is used for collision detection only;
    # CNN-mode uses cost_rays from the c_space_mapper pipeline instead.
    def velodyne_callback(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(self.environment_dim) * 10

        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break

    def odom_callback(self, od_data):
        self.last_odom = od_data

    # ---- CNN mode: receive 3D cost rays from c_space_mapper_node ----
    def cost_rays_callback(self, msg):
        """Receive cost_rays from c_space_mapper_node.

        Message layout:
          dim[0] = "beam",  size=num_beams (36), stride=nz_
          dim[1] = "theta", size=nz_ (21),     stride=1

        Each element is the distance along beam b at robot orientation θ
        to the first collision in the 3D configuration space.
        """
        num_beams = msg.layout.dim[0].size
        nz = msg.layout.dim[1].size
        data = np.array(msg.data, dtype=np.float32)
        if data.size == num_beams * nz:
            self.cost_rays_data = data.reshape(num_beams, nz)
        else:
            rospy.logwarn("cost_rays size mismatch: %d != %d*%d",
                          data.size, num_beams, nz)

    # Preprocess cost_rays to match NavRL format:
    #   - Clip min distance to 0.1 m (avoid divide-by-zero)
    #   - Invert: near obstacles → high activation (cost-like representation)
    #   - Add channel dimension: (H, V) → (1, H, V)
    def _preprocess_cost_rays(self):
        cr = np.maximum(self.cost_rays_data, 0.1)
        cr = self.max_ray_range - cr
        return cr[np.newaxis, :, :]

    # Perform an action and read a new state
    def step(self, action):
        target = False

        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
            pass
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # read velodyne laser state
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        # Wait for the first odometry message if not yet received (avoid race after reset)
        if self.last_odom is None:
            waited = 0.0
            while self.last_odom is None and waited < 2.0:
                time.sleep(0.02)
                waited += 0.02
            if self.last_odom is None:
                raise RuntimeError(
                    "No odometry received after 2 s — "
                    "check that /r1/odom is being published")

        # Calculate robot heading from odometry data
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True

        robot_state = [distance, theta, action[0], action[1]]

        if self.use_cnn:
            # CNN mode: cost_rays (from c_space_mapper) + kinematic state
            cr_proc = self._preprocess_cost_rays()
            kinematic = np.array(robot_state, dtype=np.float32)
            # Flatten for replay buffer compatibility
            state = np.concatenate([cr_proc.flatten(), kinematic])
        else:
            state = np.append(laser_state, robot_state)

        progress = self.last_distance - distance
        self.last_distance = distance

        # Build ray data for r_ss: c-space cost_rays (CNN mode) or velodyne bins
        if self.use_cnn:
            safety_data = self.cost_rays_data.flatten()
        else:
            safety_data = np.array(self.velodyne_data)

        reward = self.get_reward(
            target, collision, action, min_laser, progress,
            heading_angle=angle,
            goal_x=self.goal_x, goal_y=self.goal_y,
            odom_x=self.odom_x, odom_y=self.odom_y,
            laser_data=safety_data,
            safety_reward_weight=self.safety_reward_weight,
            speed_reward_weight=self.speed_reward_weight,
        )
        return state, reward, done, target

    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        x = 0
        y = 0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-self.boundary, self.boundary)
            y = np.random.uniform(-self.boundary, self.boundary)
            position_ok = check_pos(x, y, self.obstacles, self.boundary)
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        # object_state.pose.position.z = 0.
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # set a random goal in empty space in environment
        self.change_goal()
        # randomly scatter boxes in the environment (only for TD3.world)
        if self.world_name == "TD3.world":
            self.random_box()
        self.publish_markers([0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
            pass
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
            pass
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0]
        self.last_distance = distance   # track for progress reward

        if self.use_cnn:
            cr_proc = self._preprocess_cost_rays()
            kinematic = np.array(robot_state, dtype=np.float32)
            state = np.concatenate([cr_proc.flatten(), kinematic])
        else:
            state = np.append(laser_state, robot_state)

        return state

    def change_goal(self):
        # Place a new goal and check if its location is not on one of the obstacles
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = False

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            goal_ok = check_pos(self.goal_x, self.goal_y, self.obstacles, self.boundary)

    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y, self.obstacles, self.boundary)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser, progress=0.0,
                   heading_angle=0.0, goal_x=0.0, goal_y=0.0,
                   odom_x=0.0, odom_y=0.0, laser_data=None,
                   safety_reward_weight=1.0, speed_reward_weight=1.0):
        """Reward with velocity-direction alignment + log-average static safety.

        r_vel = unit(Pg - Pr) · Vr          (velocity toward goal)
        r_ss  = mean(log(laser_rays))       (log-average obstacle distance)
        r = speed_weight * r_vel + safety_weight * r_ss

        Args:
            target:                bool, goal reached.
            collision:             bool, collision detected.
            action:                [linear_vel, angular_vel].
            min_laser:             min laser distance (fallback for obstacle penalty).
            progress:              distance to goal reduction since last step (unused).
            heading_angle:         robot yaw in world frame [rad].
            goal_x, goal_y:        goal position in world frame.
            odom_x, odom_y:        robot position in world frame.
            laser_data:            ndarray of ray distances (c-space cost_rays or
                                   velodyne bins).  If None, falls back to min_laser.
            safety_reward_weight:  coefficient for r_ss safety reward.
            speed_reward_weight:   coefficient for r_vel speed reward.

        Reference: NavRL (Xu et al., IEEE RA-L 2025), Eqs. (8)-(9).
        """
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            # ---- r_vel: velocity alignment toward goal ----
            vx = action[0] * math.cos(heading_angle)
            vy = action[0] * math.sin(heading_angle)
            dx = goal_x - odom_x
            dy = goal_y - odom_y
            dist_to_goal = math.sqrt(dx**2 + dy**2) + 1e-8
            r_vel = (dx / dist_to_goal) * vx + (dy / dist_to_goal) * vy

            # ---- r_ss: log-average static safety over all rays ----
            if laser_data is not None and len(laser_data) > 0:
                clipped = np.maximum(laser_data, 0.1)  # prevent log(≤0)
                r_ss = np.mean(np.log(clipped))
            else:
                r3 = lambda x: 1 - x if x < 1 else 0.0
                r_ss = -r3(min_laser)

            return speed_reward_weight * r_vel + safety_reward_weight * r_ss
