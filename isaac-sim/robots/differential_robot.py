from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.api.controllers import BaseController

import numpy as np
import math

from utils.utils import quaternion_to_euler

ASSETS_ROOT_PATH = get_assets_root_path()


class DifferentialRobot:
    def __init__(
            self,
            world,
            name: str,
            prim_path: str,
            asset_path: str,
            wheel_distance: float,
            wheel_radius: float,
            max_linear_speed: float,
    ):
        """Abstract base class for differential drive robots in Isaac Sim.
        Provides generic load, initialize, control, state reading, and
        teleport methods that work for any WheeledRobot USD asset with two
        wheel joints.

        Subclasses should call super().__init__() with the robot-specific
        parameters and may override load() if the USD asset requires
        non-standard constructor arguments.

        Args:
            world: The Isaac Sim World instance the robot will be added to.
            name (str): Unique name for the robot prim within the world scene.
            prim_path (str): USD prim path at which the robot is created in
                the stage. Must be unique if multiple robots are present.
            asset_path (str): Full path to the robot USD file, typically
                constructed from the nucleus assets root path.
            wheel_distance (float): Distance between the two drive wheels
                (axle length) in metres. Used by RobotController for
                differential drive kinematics.
            wheel_radius (float): Radius of each drive wheel in metres. Used
                by RobotController to convert linear speed to wheel angular
                velocity.
            max_linear_speed (float): Hardware-limited maximum forward speed
                in m/s. Used to clamp wheel velocity commands and prevent
                the robot from exceeding its physical limits.

        Attributes:
            robot: The WheeledRobot articulation object, set by load().
            controller (RobotController): The wheel velocity controller,
                set by initialize().
        """
        self.world            = world
        self.name             = name
        self.prim_path        = prim_path
        self.asset_path       = asset_path
        self.wheel_distance   = wheel_distance
        self.wheel_radius     = wheel_radius
        self.max_linear_speed = max_linear_speed

        self.robot = None

    def load(self) -> None:
        """Instantiates the WheeledRobot from the USD asset and adds it to
        the world scene. Must be called before world.reset() and
        robot.initialize().

        The robot is spawned at the origin with a neutral orientation
        (identity quaternion [qx=0, qy=0, qz=0, qw=1]). Use teleport()
        after initialize() to set a different starting pose.

        Note:
            Subclasses that need a specific spawn position, orientation, or
            non-standard constructor arguments (e.g. usd_path vs asset_path)
            should override this method.
        """
        self.robot = WheeledRobot(
            prim_path       = self.prim_path,
            name            = self.name,
            wheel_dof_names = ["left_wheel_joint", "right_wheel_joint"],
            asset_path      = self.asset_path,
            create_robot    = True,
            position        = np.array([0.0, 0.0, 0.0]),
            orientation     = np.array([0.0, 0.0, 0.0, 1.0]),
        )
        self.world.scene.add(self.robot)

    def initialize(self) -> None:
        """Initialises the robot's physics articulation and creates the wheel
        velocity controller. Must be called after world.reset() — the physics
        simulation view does not exist before the first reset and calling this
        earlier will raise an AttributeError.
        """
        self.robot.initialize()
        self.controller = RobotController(
            wheel_radius     = self.wheel_radius,
            wheel_base       = self.wheel_distance,
            max_linear_speed = self.max_linear_speed,
        )

    def apply_action(self, command: list) -> None:
        """Converts a [v, w] body-frame command to wheel velocities and sends
        it to the physics articulation.

        Args:
            command (list[float, float]): [linear_velocity_m_s,
                angular_velocity_rad_s]. Values outside the physical limits
                are clamped by RobotController.forward().
        """
        self.robot.apply_wheel_actions(self.controller.forward(command=command))

    def stop(self) -> None:
        """Sends a zero-velocity command to both wheels, bringing the robot
        to a stop. Useful at the start of each episode after teleporting to
        clear any residual velocity from the previous episode.
        """
        self.robot.apply_wheel_actions(self.controller.forward(command=[0.0, 0.0]))

    def get_state(self) -> dict:
        """Reads the robot's current kinematic state from the physics
        simulation and returns it as a dict with all quantities expressed
        in consistent frames.

        Velocity frame conversion: Isaac Sim returns linear velocity in the
        world frame. This method projects it onto the robot's body frame
        using the current yaw so that linear_vel[0] is the true forward
        speed and linear_vel[1] is the lateral (sideways) speed.

        Orientation: the raw quaternion from Isaac Sim is in [qw, qx, qy, qz]
        order and is converted to Euler angles (roll, pitch, yaw) via
        quaternion_to_euler with inverted=True. Yaw is the third element.

        Returns:
            dict with keys:
                "position" (np.ndarray): World-frame position [x, y, z] in
                    metres, dtype float32.
                "orientation" (np.ndarray): Euler angles [roll, pitch, yaw]
                    in radians, dtype float32.
                "linear_vel" (np.ndarray): Body-frame linear velocity
                    [forward, lateral, vertical] in m/s, dtype float32.
                "angular_vel" (np.ndarray): World-frame angular velocity
                    [wx, wy, wz] in rad/s, dtype float32. wz is the yaw rate.
        """
        position, orientation_quat = self.robot.get_world_pose()
        linear_vel  = self.robot.get_linear_velocity()
        angular_vel = self.robot.get_angular_velocity()

        roll, pitch, yaw = quaternion_to_euler(orientation_quat, degrees=False, inverted=True)
        orientation = [roll, pitch, yaw]

        vx, vy = linear_vel[0], linear_vel[1]
        forward_speed = vx * math.cos(yaw) + vy * math.sin(yaw)
        lateral_speed = -vx * math.sin(yaw) + vy * math.cos(yaw)
        linear_vel    = [forward_speed, lateral_speed, float(linear_vel[2])]

        return {
            "position":    np.array(position,    dtype=np.float32),
            "orientation": np.array(orientation, dtype=np.float32),
            "linear_vel":  np.array(linear_vel,  dtype=np.float32),
            "angular_vel": np.array(angular_vel, dtype=np.float32),
        }

    def teleport(self, position: np.ndarray, yaw: float) -> None:
        """Instantly moves the robot to a given position and heading by
        directly setting its USD world pose and zeroing all velocities.

        The yaw angle is converted to a quaternion assuming zero roll and
        pitch (the robot stays flat). Isaac Sim's quaternion convention for
        this API is [qw, qx, qy, qz].

        Args:
            position (np.ndarray): Target position [x, y, z] in metres.
                z=0.0 places the robot at ground level; use a higher value
                (e.g. 0.5) and let physics settle if the exact resting height
                is uncertain.
            yaw (float): Target heading in radians, measured counter-clockwise
                from the positive X axis. Roll and pitch are set to zero.
        """
        qw = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)
        orientation = np.array([qw, 0.0, 0.0, qz])

        self.robot.set_world_pose(position=position, orientation=orientation)
        self.robot.set_linear_velocity(np.zeros(3))
        self.robot.set_angular_velocity(np.zeros(3))


class Create3Robot(DifferentialRobot):
    # Create3-specific physical constants
    _WHEEL_DISTANCE   = 0.233
    _WHEEL_RADIUS     = 0.03575
    _MAX_LINEAR_SPEED = 0.22

    def __init__(
            self,
            world,
            prim_path: str = "/World/create_3",
            position: np.ndarray | None = None,
    ):
        """Concrete differential robot implementation for the iRobot Create3.

        Inherits all control, state reading, and teleport logic from
        DifferentialRobot and only overrides load() to pass the Create3-
        specific USD path and spawn position.

        Args:
            world: The Isaac Sim World instance the robot will be added to.
            prim_path (str): USD prim path for this robot instance. Defaults
                to "/World/create_3".
            position (np.ndarray | None): Initial spawn position [x, y, z]
                in metres. If None the robot is placed at the origin and
                teleport() should be called after initialize() to set the
                actual starting pose. Defaults to None.
        """
        super().__init__(
            world            = world,
            name             = "create_3",
            prim_path        = prim_path,
            asset_path       = ASSETS_ROOT_PATH + "/Isaac/Robots/iRobot/Create3/create_3.usd",
            wheel_distance   = self._WHEEL_DISTANCE,
            wheel_radius     = self._WHEEL_RADIUS,
            max_linear_speed = self._MAX_LINEAR_SPEED,
        )
        self.position = position

    def load(self) -> None:
        """Instantiates the Create3 WheeledRobot from the nucleus USD asset
        and adds it to the world scene.

        Overrides DifferentialRobot.load() because the Create3 USD is loaded
        via the usd_path keyword (fetched from the nucleus server) rather than
        asset_path, and because the spawn position is configurable at
        construction time rather than always at the origin.

        Must be called before world.reset() and robot.initialize().
        """
        self.robot = WheeledRobot(
            prim_path       = self.prim_path,
            name            = self.name,
            wheel_dof_names = ["left_wheel_joint", "right_wheel_joint"],
            usd_path        = self.asset_path,   # nucleus path, not local file
            create_robot    = True,
            position        = self.position,
            orientation     = np.array([0.0, 0.0, 0.0, 1.0]),
        )
        self.world.scene.add(self.robot)


class RobotController(BaseController):
    def __init__(
            self,
            wheel_radius: float,
            wheel_base: float,
            max_linear_speed: float,
    ):
        """Differential drive controller that converts body-frame velocity
        commands [v, w] into individual wheel angular velocity targets.

        The kinematic model used is the standard differential drive:
            left  = (2v - w * wheel_base) / (2 * wheel_radius)
            right = (2v + w * wheel_base) / (2 * wheel_radius)

        Both outputs are clamped to ±max_wheel_vel to respect the hardware
        speed limit of the robot.

        Args:
            wheel_radius (float): Radius of each drive wheel in metres.
            wheel_base (float): Distance between the two wheels (axle length)
                in metres.
            max_linear_speed (float): Maximum robot linear speed in m/s. Used
                to derive the maximum allowable wheel angular velocity:
                max_wheel_vel = max_linear_speed / wheel_radius.
        """
        super().__init__(name="robot_controller")
        self._wheel_radius  = wheel_radius
        self._wheel_base    = wheel_base
        self._max_wheel_vel = max_linear_speed / wheel_radius

    def forward(self, command: list) -> ArticulationAction:
        """Converts a [v, w] body-frame command to an ArticulationAction
        containing individual wheel angular velocity targets.

        Args:
            command (list[float, float]): [linear_velocity_m_s,
                angular_velocity_rad_s]. Positive v moves the robot forward,
                positive w turns it counter-clockwise.

        Returns:
            ArticulationAction: Joint velocity targets for the left and right
                wheel joints, clamped to ±max_wheel_vel rad/s.
        """
        v, w  = command[0], command[1]
        left  = (2 * v - w * self._wheel_base) / (2 * self._wheel_radius)
        right = (2 * v + w * self._wheel_base) / (2 * self._wheel_radius)

        left  = np.clip(left,  -self._max_wheel_vel, self._max_wheel_vel)
        right = np.clip(right, -self._max_wheel_vel, self._max_wheel_vel)

        return ArticulationAction(joint_velocities=[left, right])