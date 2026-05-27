from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.api.controllers import BaseController

import numpy as np
import math

WHEEL_DISTANCE = 0.233
WHEEL_RADIUS = 0.03575 
MAX_LINEAR_SPEED = 0.22
ROBOT_HEIGHT = 0.138

assets_rooth_path = get_assets_root_path()

asset_path = assets_rooth_path + "/Isaac/Robots/iRobot/Create3/create_3.usd"

class Create3Robot:
    def __init__(
            self,
            world,
            prim_path: str ="/World/create_3",
            position: np.ndarray | None = None
    ):
        """The class that creates the Create3Robot.

        Args:
            world (_type_): The world in which the robot will be.
            prim_path (str, optional): The path to which the instance of the robot has to be created. Defaults to "/World/create_3".
            position (np.ndarray | None, optional): The base position of the robot. Defaults to None.
        """
        
        self.world = world
        self.prim_path = prim_path
        self.position = position
    
    def load(self):
        """The function that loads the robot into the world.
        """

        self.robot = WheeledRobot(
            prim_path=self.prim_path,
            name="create_3",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            usd_path=asset_path,
            create_robot=True,
            position=self.position,
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )

        self.world.scene.add(self.robot)
    
    def initialize(self):
        """The function that initializes the physic of the robot. It has to be called after the first reset of the world.
        """

        self.robot.initialize()
        self.controller = RobotController()
    
    def apply_action(self, command):
        """The function that give instruction for the robot to move.

        Args:
            command (tuple[float, float]): The command the robot has to follow to move.
        """

        self.robot.apply_wheel_actions(self.controller.forward(command=command))

    def stop(self):
        """The function that stops the robot.
        """

        self.robot.apply_wheel_actions(self.controller.forward(command=[0.0, 0.0]))
    
    def get_state(self):
        """The function that returns the state of the robot : its position, speeds, and yaw.

        Returns:
            state (dict): The state of the robot.
        """

        position, orientation = self.robot.get_world_pose()
        linear_vel = self.robot.get_linear_velocity()
        angular_vel = self.robot.get_angular_velocity()

        qw, qx, qy, qz = orientation

        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        vx, vy = linear_vel[0], linear_vel[1]
        forward_speed = vx * math.cos(yaw) + vy * math.sin(yaw)

        linear_vel = [forward_speed, 0.0, float(linear_vel[2])]

        state = {
            "position": np.array(position, dtype=np.float32),
            "orientation": np.array(orientation, dtype=np.float32),
            "linear_vel": np.array(linear_vel, dtype=np.float32),
            "angular_vel": np.array(angular_vel, dtype=np.float32),
            "yaw": float(yaw)
        }

        return state
    
    def teleport(self, position, yaw):
        """The function that teleport the robot to a given position with a given yaw.

        Args:
            position (tuple[float, float]): The position we want to give to the robot
            yaw (float): The yaw angle we want to give to the robot
        """

        # yaw = yaw*math.pi/180 if yaw in degrees, nothing else
        
        qw = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)
        orientation = np.array([qw, 0.0, 0.0, qz])

        self.robot.set_world_pose(position=position, orientation=orientation)
        self.robot.set_linear_velocity(np.zeros(3))
        self.robot.set_angular_velocity(np.zeros(3))
        
class RobotController(BaseController):

    def __init__(self):
        """The Controller class for our differential robot.
        """
        super().__init__(name="robot_controller")
        self._wheel_radius = WHEEL_RADIUS
        self._wheel_base = WHEEL_DISTANCE
        self._max_wheel_vel = MAX_LINEAR_SPEED / WHEEL_RADIUS
    
    def forward(self, command):
        """The function that moves the robot given a command on the wheels.

        Args:
            command (tuple[float, float]): The command for the wheels.

        Returns:
            The order on the joints with the class ArticulationAction, given our command.
        """
        v, w = command[0], command[1]
        left  = (2 * v - w * self._wheel_base) / (2 * self._wheel_radius)
        right = (2 * v + w * self._wheel_base) / (2 * self._wheel_radius)

        left  = np.clip(left,  -self._max_wheel_vel, self._max_wheel_vel)
        right = np.clip(right, -self._max_wheel_vel, self._max_wheel_vel)

        return ArticulationAction(joint_velocities=[left, right])