import numpy as np
import math

import gymnasium as gym
from gymnasium.spaces import Box

class PrexIsaacEnv(gym.Env):
    def __init__(
            self,
            max_episode_length=1000,
            max_linear_speed=0.7,
            max_angular_speed=0.4,
            radius_target=0.3,
            physics_dt=1.0/60.0,
            rendering_dt=1.0,
            verbose=False,
            clipping_limit=20.0,
            max_speed_bonus=5.0,
            repeating_action=1,
            device="cuda",
            seed=None,
            topic_sub="/prex/sensor_data",
            topic_sub_odom=None,
            topic_pub="/prex/cmd_vel",
            type_ros2_msg="Twist",
            arena_geometry=[(2.0, 2.0), 0.2, 0.5]
    ):
        super().__init__()

        # --- Hyperparameters --------------------------------------------
        self.max_episode_length = max_episode_length
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.radius_target = radius_target
        self.physics_dt = physics_dt
        self.rendering_dt = rendering_dt
        self.verbose = verbose
        self.clipping_limit = clipping_limit
        self.max_speed_bonus = max_speed_bonus
        self.repeating_action = repeating_action
        self.device = device
        self.seed = seed

        # --- ROS2 -------------------------------------------------------
        self.topic_sub = topic_sub
        self.topic_sub_odom = topic_sub_odom
        self.topic_pub = topic_pub
        self.type_ros2_msg = type_ros2_msg

        # --- Spaces -----------------------------------------------------
        self.action_space = Box(
            low=np.array([-self.max_linear_speed, -self.max_angular_speed]),
            high=np.array([ self.max_linear_speed,  self.max_angular_speed]),
            dtype=np.float32
        )

        self.observation_space = Box(
            low=np.array([-4.0, -4.0, -4.0, -4.0,
                        -self.max_linear_speed, -self.max_angular_speed,
                        -1.0, -1.0, -2.0, -math.pi, -1.0, -1.0, -100.0]),
            high=np.array([4.0, 4.0, 4.0, 4.0,
                        self.max_linear_speed, self.max_angular_speed,
                        1.0, 1.0, 2.0, math.pi, 1.0, 1.0, 100.0]),
            dtype=np.float32
        )
        
        self.state = np.zeros(13, dtype=np.float32)
        self.max_bounds = np.array(
            [self.max_linear_speed, self.max_angular_speed], dtype=np.float32
        )

        # --- Internal State ---------------------------------------------
        self.perimeter = arena_geometry[0]
        self.depth = arena_geometry[1]
        self.heigth = arena_geometry[2]
        self.goal = np.zeros(2, dtype=np.float32)
        self.dist = 0.0
        self.theta = 0.0
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.position = np.zeros(3, dtype=np.float32)
        self.step_counter = 0
        self.episode_counter = 0
        self.timestep = 0
        self.action = None
        self.info = {}
        self.delta = 0.0
        self.heading_vec = np.zeros(2, dtype=np.float32)

        # --- Isaac Sim Objects ------------------------------------------
        self.world = None
        self.arena = None
        self.robot = None
        self.sensors = None

        self.launch()
    
    def launch(self):
        
        # --- 1. Launch Simulation App -----------------------------------
        from isaacsim import SimulationApp

        self.app = SimulationApp({"headless": True,
                                  "physics_dt": self.physics_dt,
                                  "rendering_dt": self.rendering_dt
                                  }
                                )

        # --- 2. Import Isaac API ----------------------------------------
        from isaacsim.core.api import World

        from envs.arena import Arena
        from envs.robot import Create3Robot
        from envs.sensors import UltrasonicSensors

        # --- 3. Create world --------------------------------------------
        self.world = World()

        # --- 4. Build Arena ---------------------------------------------
        self.arena = Arena(
            world=self.world,
            perimeter=self.perimeter,
            depth=self.depth,
            height=self.heigth
        )
        self.arena.build()

        # --- 5. Add Robot -----------------------------------------------
        self.robot = Create3Robot(world=self.world)
        self.robot.load()

        # --- 6. Add Sensors ---------------------------------------------
        self.sensors = UltrasonicSensors(perimeter=self.perimeter)

        # --- 7. Physic Initialization -----------------------------------
        self.world.reset()
        self.robot.initialize()

        print("[PrexIsaacEnv] Isaac Sim environment ready (headless).")
    
    def spawn_robot(self):

        margin = 0.3
        
        hx = self.perimeter[0] / 2.0 - margin
        hy = self.perimeter[1] / 2.0 - margin

        spawn_x = np.random.uniform(-hx, hx)
        spawn_y = np.random.uniform(-hy, hy)
        spawn_yaw = np.random.uniform(-math.pi, math.pi)

        return spawn_x, spawn_y, spawn_yaw

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.episode_counter += 1
        self.step_counter = 0
        self.info.clear()

        spawn_x, spawn_y, spawn_yaw = self.spawn_robot()

        self.robot.teleport(position=np.array([spawn_x, spawn_y, 0.138]), yaw=spawn_yaw)
        self.robot.stop()

        for _ in range(5):
            self.world.step(render=False)

        self.read_state()
        self.prev_dist = self.dist

        if self.verbose:
            print(f"[reset] spawn=({spawn_x:.2f},{spawn_y:.2f}) yaw={spawn_yaw:.2f} dist={self.dist:.2f}")

        return self.state.copy(), self.info
    
    def step(self, action):

        self.step_counter += 1
        self.action = action
        self.prev_dist = self.dist

        linear_vel = float(np.clip(action[0], -self.max_linear_speed, self.max_linear_speed))
        angular_vel = float(np.clip(action[1], -self.max_angular_speed, self.max_angular_speed))

        for _ in range(max(1, self.repeating_action)):
            self.robot.apply_action(command=[linear_vel, angular_vel])
            self.world.step(render=False)
        
        self.read_state()
        reward, terminated, truncated = self.update_reward(self.state, action)
        self.timestep += 1

        if self.verbose:
            print(f"[step {self.step_counter}] v={linear_vel:.3f} w={angular_vel:.3f} "f"dist={self.dist:.3f} reward={reward:.3f}")

        return self.state.copy(), reward, terminated, truncated, self.info
    
    def read_state(self):

        robot_state = self.robot.get_state()

        position = robot_state["position"]
        linear_vel = robot_state["linear_vel"]
        angular_vel = robot_state["angular_vel"]
        yaw = robot_state["yaw"]

        dists = self.sensors.get_distances(position=position,
                                           yaw=yaw)

        self.heading_vec = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        pos_vec = position[:2]
        norm = np.linalg.norm(pos_vec)
        if norm > 1e-6:
            cos_delta = np.dot(pos_vec, self.heading_vec) / (norm * 1.0)
            self.delta = float(np.arccos(np.clip(cos_delta, -1.0, 1.0)))
        else:
            self.delta = 0.0
        
        self.state[0:4] = dists
        self.state[4] = linear_vel[0]
        self.state[5] = angular_vel[2]
        self.state[6:9] = position
        self.state[9] = yaw
        self.state[10:12] = self.heading_vec
        self.state[12] = self.delta

        self.theta = yaw
        self.position = position
        self.linear_speed = float(linear_vel[0])
        self.angular_speed = float(angular_vel[2])
        self.dist = float(np.linalg.norm(position[:2] - self.goal))
    
    def update_reward(self, state, action):

        terminated = False
        truncated  = False

        dist_improvement = self.prev_dist - self.dist
        reward = 10.0 * dist_improvement

        reward -= 0.01

        min_sensor = float(np.min(state[0:4]))
        if min_sensor < 0.35:
            reward -= (0.35 - min_sensor) * 5.0

        if self.dist <= self.radius_target:
            terminated = True
            self.info["terminate"] = "reached the goal"
            reward += 100.0
            return reward, terminated, truncated

        if self.position[2] < 0.05 or self.position[2] > 0.40 or self.dist > 4.0:
            terminated = True
            self.info["terminate"] = "flipped or out of bounds"
            reward = -10.0
            return reward, terminated, truncated

        if self.step_counter >= self.max_episode_length:
            truncated = True
            self.info["terminate"] = "max episode length"
            reward = -1.0
            return reward, terminated, truncated

        return reward, terminated, truncated

    def render(self):
        return None

    @property
    def max_bounds(self):
        return self._max_bounds

    @max_bounds.setter
    def max_bounds(self, v):
        self._max_bounds = v

    def close(self):
        if self.app is not None:
            self.app.close()
            print("[PrexIsaacEnv] SimulationApp closed.")

        
