import numpy as np
import math

import gymnasium as gym
from gymnasium.spaces import Box


class State:
    def __init__(
            self,
            sensors: dict,
            position: dict,
            orientation: dict,
            linear_speed: dict,
            angular_speed: dict,
            heading_vec: tuple,
            controller: bool
    ):
        """Flexible state container that builds a Gymnasium observation space
        dynamically from a declarative configuration. Each field is a dict
        mapping component names to their maximum absolute value, or None to
        exclude that component from the state vector entirely.

        Args:
            sensors (dict): Sensor readings and their maximum values.
                Keys are sensor names (e.g. "front", "back", "left", "right"),
                values are the maximum expected distance in metres, or None to
                exclude. Example: {"front": 4.0, "back": 4.0, "left": None, "right": None}
            position (dict): Robot position components and their bounds in
                metres. Keys are "x", "y", "z", values are the maximum absolute
                coordinate, or None to exclude.
                Example: {"x": 2.0, "y": 2.0, "z": None}
            orientation (dict): Robot orientation components and their bounds
                in radians. Keys are "roll", "pitch", "yaw", values are the
                maximum absolute angle, or None to exclude.
                Example: {"roll": None, "pitch": None, "yaw": math.pi}
            linear_speed (dict): Linear velocity components and their bounds in
                m/s. Keys are "vx", "vy", "vz", values are the maximum absolute
                speed, or None to exclude.
                Example: {"vx": 0.5, "vy": None, "vz": None}
            angular_speed (dict): Angular velocity components and their bounds
                in rad/s. Keys are "wx", "wy", "wz", values are the maximum
                absolute rate, or None to exclude.
                Example: {"wx": None, "wy": None, "wz": 0.4}
            heading_vec (tuple | None): Maximum absolute values for the heading
                vector components (cos_yaw, sin_yaw, delta) as a 3-tuple, or
                None to exclude the heading vector entirely.
                Example: (1.0, 1.0, 100.0)
            controller (bool | None): If True, appends a single binary dimension
                to the state vector encoding whether the obstacle avoidance
                controller was active on the current step (0 = inactive,
                1 = active). The bounds for this dimension are fixed at [0, 1]
                regardless of value. Pass False or None to exclude it.
                Example: True
            
        Attributes:
            nb_sensors (int): Number of active sensor dimensions, counted during
                init_spaces(). Used by PrexIsaacEnv.controller() to determine
                which obstacle avoidance logic to apply.

        Note:
            Call init_spaces() after construction to build the observation
            space and initialise the internal state vector.
        """
        self.sensors = sensors
        self.position = position
        self.orientation = orientation
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.heading_vec = heading_vec
        self.controller = controller

        self.state_low    = []
        self.state_high   = []
        self.state_length = 0

        self.nb_sensors = 0

    def init_spaces(self) -> None:
        """Builds the Gymnasium Box observation space and the internal state
        vector from the configuration passed to __init__.

        Iterates over each field in declaration order (sensors → position →
        orientation → linear_speed → angular_speed → heading_vec → controller).
        Components whose value is None are skipped. All included components are 
        assumed symmetric around zero, so bounds are set to [-value, +value].

        The heading_vec field is treated as a 3-element group
        (cos_yaw, sin_yaw, delta) and adds 3 dimensions at once.
        The controller field adds a single dimension if not None.

        The controller field, if true, appends one dimension with bounds
        [0, 1] rather than symmetric [-value, +value] bounds, since the flag
        is always non-negative.

        After this call the following attributes are available:
            observation_space (gymnasium.spaces.Box): The observation space.
            state (np.ndarray): Zero-initialised state vector of shape
                (state_length,) and dtype float32.
            state_length (int): Total number of active state dimensions.
            nb_sensors (int): Number of sensor dimensions included in the
                state vector (i.e. sensor entries whose value was not None).
        """
        for sensor in self.sensors:
            if self.sensors[sensor] is not None:
                self.state_low.append(-self.sensors[sensor])
                self.state_high.append(self.sensors[sensor])
                self.state_length += 1
                self.nb_sensors += 1

        for coordinate in self.position:
            if self.position[coordinate] is not None:
                self.state_low.append(-self.position[coordinate])
                self.state_high.append(self.position[coordinate])
                self.state_length += 1

        for angle in self.orientation:
            if self.orientation[angle] is not None:
                self.state_low.append(-self.orientation[angle])
                self.state_high.append(self.orientation[angle])
                self.state_length += 1

        for lin_vel in self.linear_speed:
            if self.linear_speed[lin_vel] is not None:
                self.state_low.append(-self.linear_speed[lin_vel])
                self.state_high.append(self.linear_speed[lin_vel])
                self.state_length += 1

        for ang_vel in self.angular_speed:
            if self.angular_speed[ang_vel] is not None:
                self.state_low.append(-self.angular_speed[ang_vel])
                self.state_high.append(self.angular_speed[ang_vel])
                self.state_length += 1

        if self.heading_vec is not None:
            for bound in self.heading_vec:
                self.state_low.append(-bound)
                self.state_high.append(bound)
            self.state_length += 3
        
        if self.controller:
            self.state_low.append(0)
            self.state_high.append(1)
            self.state_length += 1

        self.state_low  = np.array(self.state_low,  dtype=np.float32)
        self.state_high = np.array(self.state_high, dtype=np.float32)

        self.observation_space = Box(
            low=self.state_low,
            high=self.state_high,
            dtype=np.float32
        )
        self.state = np.zeros(self.state_length, dtype=np.float32)

    def update_values(self,
                    sensors=None,
                    position=None,
                    orientation=None,
                    linear_speed=None,
                    angular_speed=None,
                    heading_vec=None,
                    controller=None
                    ) -> None:
        """Stores the individual state components as rounded attributes.
        The flat state vector is assembled lazily by get_values() when
        needed, rather than eagerly here.

        All inputs are converted to 1D float32 numpy arrays automatically,
        so scalars, lists, tuples, and arrays of any shape are all accepted.
        Rounding to 5 decimal places on each component reduces accumulated
        floating-point noise without affecting training dynamics.

        Args:
            sensors: Sensor distance readings. Accepts any array-like of
                shape (n_sensors,) or a scalar if nb_sensors == 1.
            position: Robot world-frame position [x, y, z]. Accepts any
                array-like of length 3 or a scalar.
            orientation: Yaw angle in radians. Accepts a scalar float, a
                1-element list, or a 1D array.
            linear_speed: Forward speed in m/s. Accepts a scalar float,
                a 1-element list, or a 1D array.
            angular_speed: Yaw rate in rad/s. Accepts a scalar float,
                a 1-element list, or a 1D array.
            heading_vec: Heading vector and angular error
                [cos_yaw, sin_yaw, delta]. Accepts any array-like of
                length 3.
            controller: Binary flag indicating whether the obstacle
                avoidance controller was active on this step. Accepts a
                scalar int or bool, a 1-element list, or a 1D array.
                Value is 1 if active, 0 otherwise.
        """
        def _to_array(x):
            """Converts any scalar or array-like to a flat float32 array."""
            if x is None:
                return np.array([], dtype=np.float32)
            return np.round(
                np.atleast_1d(np.array(x, dtype=np.float32)).flatten(),
                5
            )

        self.sensors      = _to_array(sensors)
        self.position     = _to_array(position)
        self.orientation  = _to_array(orientation)
        self.linear_speed = _to_array(linear_speed)
        self.angular_speed = _to_array(angular_speed)
        self.heading_vec  = _to_array(heading_vec)
        self.controller   = _to_array(controller)

    def get_values(self) -> np.ndarray:
        """Assembles and returns the flat state vector by concatenating all
        stored components in declaration order, then writes the result into
        the pre-allocated self.state array.

        Returns:
            np.ndarray: 1D array of shape (state_length,) and dtype float32
                containing the values set by the most recent update_values()
                call. The same array object is reused each call to avoid
                repeated allocation.
        """
        new_state = np.concatenate([self.sensors,
                                    self.position,
                                    self.orientation,
                                    self.linear_speed,
                                    self.angular_speed,
                                    self.heading_vec,
                                    self.controller])
        
        for i, value in enumerate(new_state):
            self.state[i] = value
        return self.state

    def get_observation_space(self) -> Box:
        """Returns the Gymnasium observation space built by init_spaces().

        Returns:
            gymnasium.spaces.Box: The observation space with bounds derived
                from the configuration passed to __init__.
        """
        return self.observation_space


class PrexIsaacEnv(gym.Env):
    def __init__(
            self,
            max_episode_length: int = 1000,
            max_linear_speed: float = 0.7,
            max_angular_speed: float = 0.4,
            target_point: tuple = (0.0, 0.0),
            radius_target: float = 0.3,
            physics_dt: float = 1.0 / 60.0,
            rendering_dt: float = 1.0,
            verbose: bool = False,
            ppo: bool = False,
            cube: int = 0,
            cube_dimension: float = 0.3,
            dist_objects: float = 0.2,
            borderless_perimeter: list = (5.0, 5.0),
            arena_geometry: list = [(2.0, 2.0), 0.2, 0.5],
            arena: bool = False,
            sensor_config = (180.0, 15.0, 5),
            activate_controller: bool = True,
            repeating_action: int = 1,
            device: str = "cuda",
            seed: int = None,
    ):
        """Gymnasium-compatible RL environment for the Create3 robot navigating
        a rectangular arena in Isaac Sim. Supports both SAC (custom) and PPO
        (Stable Baselines 3) training pipelines. Optionally spawns a cube
        obstacle or enables analytic ultrasonic sensors.

        The environment launches Isaac Sim in headless mode on construction,
        builds the arena, loads the robot, and initialises physics. Episodes
        end when the robot reaches the goal, leaves the arena, flips over, or
        exhausts the step budget.

        Args:
            max_episode_length (int): Maximum number of steps per episode
                before truncation. Defaults to 1000.
            max_linear_speed (float): Maximum linear speed of the robot in m/s,
                used to clip actions and define observation bounds.
                Defaults to 0.7.
            max_angular_speed (float): Maximum angular speed of the robot in
                rad/s, used to clip actions and define observation bounds.
                Defaults to 0.4.
            target_point (tuple): World-frame 2D coordinates [x, y] of the goal
                that the robot must reach. Defaults to (0.0, 0.0) — the origin.
            radius_target (float): Radius in metres within which the robot is
                considered to have reached the goal. Defaults to 0.3.
            physics_dt (float): Physics simulation timestep in seconds.
                Defaults to 1/60.
            rendering_dt (float): Rendering timestep in seconds. Set to a
                large value (e.g. 1.0) to effectively disable rendering during
                headless training. Defaults to 1.0.
            verbose (bool): If True, prints step-level debug information to
                stdout. Defaults to False.
            ppo (bool): If True, uses the PPO reward formulation instead of
                the SAC one. Defaults to False.
            cube (int): Number of cube obstacles to spawn in the arena. Each
                cube is repositioned at the start of every episode. Set to 0
                for no obstacles. Stored as self.nb_cube. Defaults to 0.
            cube_dimension (float): Side length of each cube obstacle in metres.
                Defaults to 0.3.
            dist_objects (float): Additional clearance in metres added to the
                minimum separation between all spawned objects (goal, robot,
                cubes). Increase to spread obstacles further apart.
                Defaults to 0.2.
            borderless_perimeter (list): Spawn region dimensions [width, height]
                in metres used when arena=False. The robot and cubes are confined
                to this region even without physical walls. Defaults to (5.0, 5.0).
            arena_geometry (list): Arena geometry as
                [(width, height), wall_depth, wall_height] in metres.
                Defaults to [(2.0, 2.0), 0.2, 0.5].
            arena (bool): If True, builds rectangular arena walls around the
                training area. If False, only a ground plane is created and
                self.perimeter is set to a large default value (5.0 x 5.0 m)
                to allow open-world training. Defaults to False.
            sensor_config (tuple): Configuration passed to UltrasonicSensors
                describing the sensor cone geometry as (total_angle_deg,
                cone_angle_deg, n_rays_per_sensor). Defaults to (180.0, 15.0, 5).
            activate_controller (bool): If True, the obstacle avoidance safety
                controller is applied to every action in step() before it reaches
                the robot. Set to False to disable the override entirely, e.g.
                when evaluating a policy that has already learned to avoid
                obstacles. Defaults to True.
            repeating_action (int): Number of physics steps to simulate per
                call to step(). Higher values give the robot more time to
                respond to each action command. Defaults to 1.
            device (str): Torch device for SAC training ("cuda" or "cpu").
                Defaults to "cuda".
            seed (int | None): Random seed passed to the Gymnasium base class.
                Defaults to None.

        Raises:
            AssertionError: If both cube=True and sensors=True are passed,
                since the two modes are mutually exclusive.
        
        The following internal state attributes are also initialised:
            self.orientation (np.ndarray): Current Euler angles [roll, pitch, yaw]
                in radians, shape (3,). Updated each step by read_state().
            self.last_position (np.ndarray): Robot position from the previous
                step, shape (3,). Stored for reward shaping or debugging.
            self.controlled_speed (float): Magnitude of the linear velocity
                at the moment the controller last triggered an override. Stored
                for reward shaping purposes.
            self.rend (bool): Derived from rendering_dt. True when
                rendering_dt < 1.0, indicating that the viewport should be
                updated during step(). Used internally to pass render=True to
                sensors.draw_rays().
            self.needed_control (bool): Flag set to True by controller() when the
                obstacle avoidance override was triggered on the current step.
                Used by update_reward() to apply a penalty. Reset externally or
                by the caller between steps.
        """
        super().__init__()

        self.max_episode_length = max_episode_length
        self.max_linear_speed   = max_linear_speed
        self.max_angular_speed  = max_angular_speed
        self.target_point = target_point
        self.radius_target      = radius_target
        self.physics_dt         = physics_dt
        self.rendering_dt       = rendering_dt
        self.verbose            = verbose
        self.ppo                = ppo
        self.nb_cube            = cube
        self.cube_dimension = cube_dimension
        self.dist_objects       = dist_objects
        self.borderless_perimeter = borderless_perimeter
        self.has_arena          = arena
        self.sensor_config = sensor_config
        self.activate_controller = activate_controller
        self.repeating_action   = repeating_action
        self.device             = device
        self.seed               = seed

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
                "front": 6.0, "back": 6.0,
                "left":  6.0, "right": 6.0,
            },
            position={
                "x": np.inf, "y": np.inf, "z": np.inf,
            },
            orientation={
                "roll": None, "pitch": None, "yaw": math.pi,
            },
            linear_speed={
                "vx": self.max_linear_speed, "vy": None, "vz": None,
            },
            angular_speed={
                "wx": None, "wy": None, "wz": self.max_angular_speed,
            },
            heading_vec=(1.0, 1.0, math.pi),
            controller=True
        )
        self.state.init_spaces()
        self.observation_space = self.state.get_observation_space()

        self.goal            = np.array(self.target_point)
        self.dist            = 0.0
        self.theta           = 0.0
        self.linear_speed    = 0.0
        self.angular_speed   = 0.0
        self.position        = np.zeros(3,  dtype=np.float32)
        self.orientation     = np.zeros(3,  dtype=np.float32)
        self.step_counter    = 0
        self.episode_counter = 0
        self.timestep        = 0
        self.action          = None
        self.info            = {}
        self.delta           = 0.0
        self.heading_vec     = np.zeros(2,  dtype=np.float32)
        self.last_action     = np.zeros(2,  dtype=np.float32)
        self.last_position   = np.zeros(3,  dtype=np.float32)

        if self.has_arena:
            self.perimeter       = arena_geometry[0]
            self.depth           = arena_geometry[1]
            self.heigth          = arena_geometry[2]
        else:
            self.perimeter = np.array(self.borderless_perimeter)

        if self.rendering_dt < 1.0:
            self.rend = True
        else:
            self.rend = False

        self.world   = None
        self.arena   = None
        self.robot   = None
        self.sensors = None
        self.cubes    = None

        self.needed_control = False
        self.controlled_speed = 0.0

        self.__launch()

    def __launch(self) -> None:
        """Launches Isaac Sim in headless mode, builds the scene, and
        initialises the physics simulation. Called once automatically by
        __init__. The double underscore prefix makes this method name-mangled
        — it cannot be called or overridden by subclasses or external code.

        Steps performed:
            1. Start SimulationApp (headless).
            2. Import Isaac Sim API (must happen after SimulationApp starts).
            3. Create the World and add a default ground plane.
            4. Optionally build arena walls (only if has_arena=True).
            5. Instantiate the Cube manager and create nb_cube cubes at the
               origin. If nb_cube=0 no visible cubes are added to the scene
               but the manager object is still created.
            6. Load the Create3 robot.
            7. Initialise the UltrasonicSensors with the nb_sensors count
               derived from the State configuration and the sensor_config
               cone geometry tuple.
            8. Run world.reset(), robot.initialize(), and
               sensors.initialize() to start physics and the PhysX query
               interface.
            9. Run 30 warm-up steps so physics settles before training.
        """
        from isaacsim import SimulationApp

        self.app = SimulationApp({
            "headless":     True,
            "physics_dt":   self.physics_dt,
            "rendering_dt": self.rendering_dt,
        })

        from isaacsim.core.api import World
        from envs.isaacsim_elements.arena import Arena
        from robots.differential_robot import Create3Robot
        from robots.sensors.ultrasonic_sensors import UltrasonicSensors
        from envs.isaacsim_elements.cube import Cube

        self.world = World()
                
        self.world.scene.add_default_ground_plane()

        if self.has_arena:
            self.arena = Arena(
                world=self.world,
                perimeter=self.perimeter,
                depth=self.depth,
                height=self.heigth,
            )
            self.arena.build()
 
        self.cubes = Cube(
            world=self.world,
            dimension=self.cube_dimension,
            perimeter=self.perimeter,
        )
        self.cubes.create_cubes(nb_cubes=self.nb_cube)

        self.robot = Create3Robot(world=self.world, prim_path="/World/create_3")
        self.robot.load()

        self.sensors = UltrasonicSensors(nb_sensors=self.state.nb_sensors, 
                                         sensor_config=self.sensor_config)

        self.world.reset()
        self.robot.initialize()
        self.sensors.initialize(render=self.rend)

        for _ in range(30):
            self.world.step(render=False)

        print("[PrexIsaacEnv] Isaac Sim environment ready (headless).")

    def spawn_robot_random_pos(self) -> None:
        """Teleports the robot to a random position and orientation inside the
        arena, keeping a safety margin from the walls to avoid spawning inside
        or directly against them.

        The spawn region is the arena interior shrunk by a fixed margin of 0.3 m
        on each side. Yaw is sampled uniformly from [-π, π]. The robot is
        teleported to z=0.0 and the velocity controller is reset to zero.

        If verbose=True, prints the spawn coordinates and initial distance to
        the goal.
        """
        margin = 0.3
        hx = self.perimeter[0] / 2.0 - margin
        hy = self.perimeter[1] / 2.0 - margin

        spawn_x   = np.random.uniform(-hx, hx)
        spawn_y   = np.random.uniform(-hy, hy)
        spawn_yaw = np.random.uniform(-math.pi, math.pi)

        self.robot.teleport(
            position=np.array([spawn_x, spawn_y, 0.0]),
            yaw=spawn_yaw
        )
        self.robot.stop()

        if self.verbose:
            print(f"[reset] spawn=({spawn_x:.2f},{spawn_y:.2f}) "
                  f"yaw={spawn_yaw:.2f} dist={self.dist:.2f}")

    def reset(self, seed: int = None, options: dict = None):
        """Resets the environment for a new episode. Increments the episode
        counter, clears episode info, respawns the robot at a random position,
        optionally repositions the cube obstacle, and reads the initial state.

        After spawning the robot, all cube obstacles are repositioned by
        set_up_all_cubes(), which places each cube at a random location that
        avoids the goal area, the robot, and all previously placed cubes.

        Args:
            seed (int | None): Random seed forwarded to the Gymnasium base
                class for reproducibility. Defaults to None.
            options (dict | None): Unused. Present for Gymnasium API
                compatibility. Defaults to None.

        Returns:
            tuple[np.ndarray, dict]:
                - obs (np.ndarray): Initial observation of shape
                    (state_length,) and dtype float32.
                - info (dict): Empty info dict at episode start.
        """
        super().reset(seed=seed)

        self.episode_counter += 1
        self.step_counter    = 0
        self.info.clear()

        self.spawn_robot_random_pos()
        self.last_action = np.zeros(2)

        self.read_state()
        self.prev_dist = self.dist

        self.cubes.set_up_all_cubes(
            target_position=self.goal,
            target_radius=self.radius_target,
            robot_position=self.position[:2],
            robot_size=0.4,
            distance_between_objects=self.dist_objects,
            nb_cubes=self.nb_cube
        )

        return self.state.get_values().copy(), self.info

    def step(self, action: np.ndarray, render: bool = False):
        """Advances the simulation by one environment step. The raw action is
        first passed through the obstacle avoidance controller, which may
        override the linear velocity component if a sensor detects an obstacle
        within the detection threshold. The (possibly modified) action is then
        applied to the robot, physics is stepped repeating_action times, the
        new state is read, and the reward and termination flags are computed.

        Args:
            action (np.ndarray): 1D array [linear_velocity, angular_velocity]
                in m/s and rad/s respectively. Values are clipped to
                [-max_linear_speed, max_linear_speed] and
                [-max_angular_speed, max_angular_speed] after the controller
                has processed them.
            render (bool): If True, passes render=True to world.step() so the
                Isaac Sim viewport is updated. Set to True when recording
                videos; leave False during training for speed. Defaults to
                False.

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]:
                - obs (np.ndarray): New observation of shape (state_length,).
                - reward (float): Scalar reward for this transition.
                - terminated (bool): True if the episode ended due to reaching
                    the goal, flipping, or leaving the arena.
                - truncated (bool): True if the episode ended due to exceeding
                    max_episode_length.
                - info (dict): Contains "terminate" key with a human-readable
                    reason string when terminated or truncated.
        
        Note:
            draw_rays() is called before world.step() so the ray visualisation
            reflects the robot's position at the start of the current step
            rather than one step behind. self.rend controls whether rendering
            is active based on the rendering_dt passed at construction.
        """
        self.step_counter += 1
        self.needed_control = False
        
        if self.activate_controller:
            action = self.controller(action)

        self.action       = action
        self.last_action  = action
        self.last_position = self.position
        self.prev_dist    = self.dist

        linear_vel  = float(np.clip(action[0], -self.max_linear_speed,  self.max_linear_speed))
        angular_vel = float(np.clip(action[1], -self.max_angular_speed, self.max_angular_speed))

        self.robot.apply_action(command=[linear_vel, angular_vel])

        if render:
            self.sensors.draw_rays(origin=self.position, yaw=self.theta)

        for _ in range(max(1, self.repeating_action)):
            self.world.step(render=render)

        self.read_state()
        reward, terminated, truncated = self.update_reward(action)
        self.timestep += 1

        if self.verbose:
            print(f"[step {self.step_counter}] v={linear_vel:.3f} "
                  f"w={angular_vel:.3f} dist={self.dist:.3f} "
                  f"reward={reward:.3f}")

        return self.state.get_values().copy(), reward, terminated, truncated, self.info

    def read_state(self) -> None:
        """Reads the robot's current physical state from Isaac Sim, computes
        derived quantities (heading vector, angular error delta), and updates
        the internal State object.

        The full set of four sensor distances (front, back, left, right) is
        always computed analytically, but only the subset declared as non-None
        in the State configuration is passed to update_state().

        The heading vector is negated relative to the robot's forward direction:
            heading_vec = [-cos(yaw), -sin(yaw)]
        This points from the robot toward the origin (goal) when the robot faces
        it, making delta a direct measure of how misaligned the robot is with the 
        goal direction.

        The state vector is assembled in this order:
            [d_s0, ..., d_sN,            ← all active sensor distances
            x, y, z,                   ← position
            yaw,                       ← orientation (yaw only)
            vx,                        ← forward linear speed
            wz,                        ← yaw rate
            cos_yaw, sin_yaw, delta,   ← heading vector + angle error
            controller_flag]           ← 1 if controller was active, else 0

        The number of sensor dimensions matches self.state.nb_sensors, which is
        determined by the State configuration in __init__ and passed to
        UltrasonicSensors at construction.

        Updates:
            self.position, self.orientation, self.theta, self.linear_speed,
            self.angular_speed, self.dist, self.heading_vec, self.delta, and
            the internal State vector via State.update_state().

        Note:
            full_sensors computes all four distances but only the first two
            (front and back) are passed to update_state(), matching the State
            configuration. The variable front (shape (1,)) is computed but
            unused — it is a leftover from the single-sensor configuration
            and can be removed.
        """
        robot_state = self.robot.get_state()

        self.position     = position = robot_state["position"]
        orientation       = robot_state["orientation"]
        linear_vel        = robot_state["linear_vel"]
        angular_vel       = robot_state["angular_vel"]
        self.theta        = yaw = float(orientation[2])
        self.linear_speed = float(linear_vel[0])
        self.angular_speed = float(angular_vel[2])
        self.dist         = float(np.linalg.norm(position[:2] - self.goal))

        self.heading_vec = np.array(
            [-math.cos(yaw), -math.sin(yaw)], dtype=np.float32
        )
        pos_vec      = position[:2]
        norm_pos_vec = np.linalg.norm(pos_vec)
        if norm_pos_vec > 1e-6:
            cos_delta  = np.dot(pos_vec, self.heading_vec) / norm_pos_vec
            self.delta = float(np.arccos(np.clip(cos_delta, -1.0, 1.0)))
        else:
            self.delta = 0.0

        full_heading = np.concatenate([self.heading_vec, [self.delta]])
        full_sensors = self.sensors.get_distances(self.position, self.theta)

        self.state.update_values(sensors=full_sensors,
                                position=self.position,
                                orientation=self.theta,
                                linear_speed=self.linear_speed,
                                angular_speed=self.angular_speed,
                                heading_vec=full_heading,
                                controller=self.needed_control
                                )

    def update_reward(
            self,
            action: np.ndarray,
            state: np.ndarray = None,
    ) -> tuple[float, bool, bool]:
        """Computes the reward and episode termination flags for the current
        transition.

        Reward formulations:
            PPO: 1/(delta + 0.3) + 1/(dist + 0.01) — always positive,
                peaks when both angle error and distance are small.
            SAC: A multiplicative term combining forward speed, heading
                alignment, and proximity to the goal:
                    (speed_term * delta_term * dist_term) / 1000
                        - int(needed_control)
                where:
                    speed_term = 1 / (0.51 - action[0])  — rewards higher
                        forward speed; 0.51 is slightly above max_linear_speed
                        to keep the term positive.
                    delta_term = 1 / (delta + 0.1)  — rewards facing the goal.
                    dist_term  = 1 / (dist + 0.1)   — rewards proximity.
                The controller penalty subtracts 1.0 when the safety
                controller overrode the action.

        Additional bonuses and overrides (applied after base reward):
            - Front sensor reading > 0.30 m (no imminent obstacle): +0.5
              bonus, rewarding the robot for maintaining safe clearance.
            - Timeout (step_counter >= max_episode_length): reward = -0.5,
              truncated = True.
            - Flipped or out of bounds (z > 0.40 or dist > 100.0):
              reward = -0.5, terminated = True.
            - Goal reached (dist <= radius_target AND action[0] >= 0.0):
              reward = 10.0, terminated = True. The forward-speed condition
              prevents the episode from ending while the robot is reversing
              through the goal.

        Note:
            The out-of-bounds threshold is 100.0 m (not 4.0 m as in earlier
            versions) to accommodate the larger open-world training area.
            The goal check requires action[0] >= 0.0 so the robot must be
            moving forward or stopped when it reaches the goal.

        Args:
            action (np.ndarray): Current action [v, w] used in the SAC
                speed term and the goal forward-speed check.
            state (np.ndarray | None): Unused. Present for interface
                compatibility with callers that pass a state argument.
                Defaults to None.

        Returns:
            tuple[float, bool, bool]:
                - reward (float): Scalar reward for this transition.
                - terminated (bool): True if the episode ended definitively.
                - truncated (bool): True if the episode ended due to timeout.
        """
        terminated = False
        truncated  = False

        if self.ppo:
            reward = 1 / (self.delta + 0.3) + 1 / (self.dist + 0.01)
        else:
            speed_term = 1 / (0.51 - action[0])
            delta_term = 1 / (self.delta + 0.1)
            dist_term = 1 / (self.dist + 0.1)
            reward = (speed_term * delta_term * dist_term) / 1000 - int(self.needed_control)

        dist = self.state.sensors[0] 
        if dist > 0.30:
            reward += 0.5

        if self.step_counter >= self.max_episode_length:
            truncated = True
            self.info["terminate"] = "max episode length"
            reward = -0.5

        if self.position[2] > 0.40 or self.dist > 100.0:
            terminated = True
            self.info["terminate"] = "flipped or out of bounds"
            reward = -0.5

        if self.dist <= self.radius_target and action[0] >= 0.0:
            terminated = True
            self.info["terminate"] = "reached the goal"
            reward = 10.0

        return reward, terminated, truncated

    def controller(self, action: np.ndarray, noDetectionDist: float = 0.40) -> np.ndarray:
        """Obstacle avoidance safety controller that overrides the linear
        velocity component of the action when a sensor detects an obstacle
        within the detection threshold.

        When an override occurs, self.needed_control is set to True and
        self.controlled_speed is set to the absolute value of the linear
        velocity that was suppressed. self.controlled_speed can be used by
        update_reward() or external logging to measure how aggressively the
        controller intervened.

        The behaviour depends on how many sensors are active (self.state.nb_sensors):
            0 sensors: the action is returned unchanged and a warning is printed
                if verbose=True.
            1 sensor (front only): linear override is triggered when the robot
                moves forward (lin_vel > 0) and the front sensor reads below
                noDetectionDist.
            2+ sensors (front and back): linear override is triggered when the
                robot moves forward into a front obstacle or backward into a
                rear obstacle.

        Args:
            action (np.ndarray): Raw action [linear_velocity, angular_velocity]
                from the policy, in m/s and rad/s.
            noDetectionDist (float): Distance threshold in metres below which a
                sensor reading is considered an obstacle detection. Defaults to
                0.50.

        Returns:
            np.ndarray: Possibly modified action [linear_velocity,
                angular_velocity]. If the controller overrides, linear_velocity
                is set to 0.0 and angular_velocity is preserved unchanged.
                If no obstacle is detected, the original action is returned
                unmodified.
        """
        lin_vel, ang_vel = action
        controlled_action = np.array([0.0, ang_vel])
        nb_sensors = self.state.nb_sensors
        state = self.state.get_values()

        if nb_sensors == 0:
            if self.verbose:
                print("No sensors, cannot use the controller.")
            return action
        
        elif nb_sensors == 1:
            if lin_vel > 0.0 and state[0] < noDetectionDist:
                if self.verbose:
                    print("Obstacle detected, controller activated")
                self.needed_control = True
                self.controlled_speed = abs(lin_vel)
                return controlled_action
            else:
                return action
        
        else:
            if (lin_vel > 0.0 and state[0] < noDetectionDist) or (lin_vel < 0.0 and state[1] < noDetectionDist):
                if self.verbose:
                    print("Obstacle detected, controller activated")
                self.needed_control = True
                self.controlled_speed = abs(lin_vel)
                return controlled_action
            else:
                return action
    
    def render(self) -> None:
        """Rendering is handled directly by Isaac Sim via world.step(render=True)
        in step() and record scripts. This method exists solely for Gymnasium
        API compatibility and does nothing.

        Returns:
            None
        """
        return None

    @property
    def max_bounds(self) -> np.ndarray:
        """Maximum action bounds as a numpy array [max_linear_speed,
        max_angular_speed]. Used by the SAC training loop to scale random
        exploration actions.

        Returns:
            np.ndarray: Array of shape (2,) with the maximum absolute values
                for linear and angular velocity.
        """
        return self._max_bounds

    @max_bounds.setter
    def max_bounds(self, v: np.ndarray) -> None:
        self._max_bounds = v

    def close(self) -> None:
        """Closes the Isaac Sim application and releases all resources. Should
        be called at the end of training or evaluation to avoid resource leaks.

        If the SimulationApp was never successfully started (e.g. due to an
        error in launch()), this method does nothing.
        """
        if self.app is not None:
            self.app.close()
            print("[PrexIsaacEnv] SimulationApp closed.")