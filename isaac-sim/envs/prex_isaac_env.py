import numpy as np
import math

import gymnasium as gym
from gymnasium.spaces import Box


class State:
    def __init__(
            self,
            sensors,
            position,
            orientation,
            linear_speed,
            angular_speed,
            heading_vec,
    ):
        
        self.sensors = sensors
        self.position = position
        self.orientation = orientation
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.heading_vec = heading_vec

        self.state_low = []
        self.state_high = []
        self.state_length = 0

    def init_spaces(self):

        for sensor in self.sensors:
            if self.sensors[sensor] != None:
                self.state_low.append(-self.sensors[sensor])
                self.state_high.append(self.sensors[sensor])
                self.state_length += 1

        for coordinate in self.position:
            if self.position[coordinate] != None:
                self.state_low.append(-self.position[coordinate])
                self.state_high.append(self.position[coordinate])
                self.state_length += 1
        
        for angle in self.orientation:
            if self.orientation[angle] != None:
                self.state_low.append(-self.orientation[angle])
                self.state_high.append(self.orientation[angle])
                self.state_length += 1
        
        for lin_vel in self.linear_speed:
            if self.linear_speed[lin_vel] != None:
                self.state_low.append(-self.linear_speed[lin_vel])
                self.state_high.append(self.linear_speed[lin_vel])
                self.state_length += 1
        
        for ang_vel in self.angular_speed:
            if self.angular_speed[ang_vel] != None:
                self.state_low.append(-self.angular_speed[ang_vel])
                self.state_high.append(self.angular_speed[ang_vel])
                self.state_length += 1
        
        if self.heading_vec != None:
            self.state_low.append(-self.heading_vec)
            self.state_high.append(self.heading_vec)
            self.state_length += 3
        
        self.state_low = np.array(self.state_low)
        self.state_high = np.array(self.state_high)
    
        self.observation_space = Box(low=self.state_low,
                                high=self.state_high,
                                dtype=np.float32)
        
        self.state = np.zeros(self.state_length, dtype=np.float32)

    def update_state(self, new_state):

        state_index = 0
        for state_element in new_state:
            self.state[state_index] = state_element
            state_index += 1
        
        self.state = np.round(self.state, 5)
        
    def get_state(self):
        return self.state
    
    def get_observation_space(self):
        return self.observation_space
        

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
            ppo=False,
            cube=False,
            sensors=False,
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
        self.ppo = ppo
        self.cube = cube
        self.has_sensors=sensors
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
            
        self.max_bounds = np.array(
            [self.max_linear_speed, self.max_angular_speed], dtype=np.float32
        )

        self.state = State(
            sensors={
                "front": 4.0,
                "back": 4.0,
                "left": 4.0,
                "right": 4.0,
            },
            position={
                "x": 2.0,
                "y": 2.0,
                "z": 2.0,
            },
            orientation={
                "roll": None,
                "pitch": None,
                "yaw": math.pi,
            },
            linear_speed={
                "vx": self.max_linear_speed,
                "vy": None,
                "vz": None,
            },
            angular_speed={
                "wx": None,
                "wy": None,
                "wz": self.max_angular_speed,
            },
            heading_vec=(1.0, 1.0, 100.0),
        )
        self.state.init_spaces()

        self.observation_space = self.state.get_observation_space()

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
        self.last_action = np.zeros(2, dtype=np.float32)

        # --- Isaac Sim Objects ------------------------------------------
        self.world = None
        self.arena = None
        self.robot = None
        self.sensors = None

        assert not self.cube or not self.sensors, "You have to choose between spawning a cube or using the four sensors"

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

        from envs.isaacsim_elements.arena import Arena
        from robots.differential_robot import Create3Robot
        from envs.isaacsim_elements.sensors import UltrasonicSensors

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

        # --- 5. Create Cube ----------------------------------------------
        if self.cube:
            from envs.isaacsim_elements.cube import Cube

            self.cube = Cube(
            world=self.world,
            scale=(0.3, 0.3, 0.3),
            perimeter=self.perimeter
            )
            self.cube.create_cube()

        # --- 6. Add Robot -----------------------------------------------
        self.robot = Create3Robot(world=self.world)
        self.robot.load()

        # --- 7. Add Sensors ---------------------------------------------
        self.sensors = UltrasonicSensors(perimeter=self.perimeter)

        # --- 8. Physic Initialization -----------------------------------
        self.world.reset()
        self.robot.initialize()

        for _ in range(30):
            self.world.step(render=False)
    
    def spawn_robot_random_pos(self):

        margin = 0.3
        
        hx = self.perimeter[0] / 2.0 - margin
        hy = self.perimeter[1] / 2.0 - margin

        spawn_x = np.random.uniform(-hx, hx)
        spawn_y = np.random.uniform(-hy, hy)
        spawn_yaw = np.random.uniform(-math.pi, math.pi)

        self.robot.teleport(position=np.array([spawn_x, spawn_y, 0.0]), yaw=spawn_yaw)
        self.robot.stop()

        if self.verbose:
            print(f"[reset] spawn=({spawn_x:.2f},{spawn_y:.2f}) yaw={spawn_yaw:.2f} dist={self.dist:.2f}")

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.episode_counter += 1
        self.step_counter = 0
        self.info.clear()

        self.spawn_robot_random_pos()

        self.last_action = np.zeros(2)

        self.read_state()
        self.prev_dist = self.dist

        # --- Spawning the cube -------------------------------------------
        if self.cube:
            robot_position = self.position[:2]

            self.cube.teleport_cube(
                target_radius=self.radius_target,
                robot_position=robot_position,
                robot_size=0.4
            )

        return self.state.copy(), self.info
    
    def step(self, action, render=False):

        self.step_counter += 1
        self.action = action
        self.last_action = action
        self.prev_dist = self.dist

        linear_vel = float(np.clip(action[0], -self.max_linear_speed, self.max_linear_speed))
        angular_vel = float(np.clip(action[1], -self.max_angular_speed, self.max_angular_speed))
        
        self.robot.apply_action(command=[linear_vel, angular_vel])
        for _ in range(max(1, self.repeating_action)):
            self.world.step(render=render)
        
        self.read_state()
        reward, terminated, truncated = self.update_reward(self.state, action)
        self.timestep += 1

        if self.verbose:
            print(f"[step {self.step_counter}] v={linear_vel:.3f} w={angular_vel:.3f} "f"dist={self.dist:.3f} reward={reward:.3f}")

        return self.state.copy(), reward, terminated, truncated, self.info
    
    def read_state(self):

        robot_state = self.robot.get_state()

        self.position = position = robot_state["position"]
        linear_vel = robot_state["linear_vel"]
        angular_vel = robot_state["angular_vel"]
        self.theta = yaw = robot_state["yaw"]

        self.linear_speed = float(linear_vel[0])
        self.angular_speed = float(angular_vel[2])
        self.dist = float(np.linalg.norm(position[:2] - self.goal))

        self.heading_vec = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        pos_vec = position[:2]
        norm_pos_vec = np.linalg.norm(pos_vec)
        if norm_pos_vec > 1e-6:
            cos_delta = np.dot(pos_vec, self.heading_vec) / (norm_pos_vec * 1.0)
            self.delta = float(np.arccos(np.clip(cos_delta, -1.0, 1.0)))
        else:
            self.delta = 0.0

        new_state = np.array([])

        # --- 1/ Sensors ---
        sensors_dists = self.sensors.get_distances(position, yaw)
        new_state = np.append(new_state, sensors_dists)

        # --- 2/ Position ---
        new_state = np.append(new_state, self.position)

        # --- 3/ Orientation ---
        new_state = np.append(new_state, self.theta)

        # --- 4/ Linear Speed ---
        new_state = np.append(new_state, self.linear_speed)

        # --- 5/ Angular Speed ---
        new_state = np.append(new_state, self.angular_speed)

        # --- 6/ Heading Vector and Delta ---
        new_state = np.append(new_state, self.heading_vec)
        new_state = np.append(new_state, self.delta)

        self.state.update_state(new_state)
    
    def update_reward(self, state, action):

        terminated = False
        truncated  = False

        if self.ppo:
            reward = 1 / (self.delta + 0.3) + 1 / (self.dist + 0.01) 
        else:
            reward = - self.delta - self.dist  

        if self.cube:
            if self.state[0] < 0.35:
                reward -= 0.05

        if self.step_counter >= self.max_episode_length:
            truncated = True
            self.info["terminate"] = "max episode length"
            reward = -1.0

        if self.position[2] > 0.40 or self.dist > 4.0:
            terminated = True
            self.info["terminate"] = "flipped or out of bounds"
            reward = -5.0

        if self.dist <= self.radius_target:
            terminated = True
            self.info["terminate"] = "reached the goal"
            reward += 100.0

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

        
