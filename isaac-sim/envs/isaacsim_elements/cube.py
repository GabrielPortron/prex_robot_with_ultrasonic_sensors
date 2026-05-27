from isaacsim.core.api.objects import FixedCuboid

import numpy as np
import math

class Cube:
    def __init__(
            self,
            world,
            scale: tuple[float, float, float] =(0.2, 0.2, 0.2),
            perimeter: tuple[float, float] =(2.0, 2.0)
    ):
        """The class that spawns a cubic obstacle for the robot's training.

        Args:
            world : The world in which we sapwn the cube
            scale (tuple[float, float, float], optional): The scale of the cube. Defaults to (0.2, 0.2, 0.2).
            perimeter (tuple[float, float], optional): The dimensions of the arena in which we spawn the cube. Defaults to (2.0, 2.0).
        """

        self.world = world
        self.scale = scale
        self.size = self.scale[2]

        self.perimeter = perimeter
        self.length, self.width = self.perimeter

        self.hx = (self.length / 2.0) - self.size / 2.0
        self.hy = (self.width / 2.0) - self.size / 2.0
    
    def create_cube(self):
        """The function that create the cube and spanws it in the middle of the world.
        """

        prim_path = "/World/Square_Arena/cube"

        self.cube = FixedCuboid(
                prim_path=prim_path,
                name="cube",
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                scale=self.scale
            )

        self.world.scene.add(self.cube)

    def teleport_cube(self, target_radius, robot_position, robot_size):
        """The function that teleports the cube in a random location, avoiding, the target area and the robot.

        Args:
            target_radius (float): The radius of the target area for the robot.
            robot_position (tuple[float, float]): The current position of the robot.
            robot_size (float): The size of the robot.

        Returns:
            _type_: _description_
        """

        cube_yaw = np.random.uniform(-math.pi, math.pi)
        cube_orientation = np.array([math.cos(cube_yaw/2.0), 0.0, 0.0, math.sin(cube_yaw/2.0)])
        
        has_valid_location = False

        while not has_valid_location:
            x = np.random.uniform(-self.hx, self.hx)
            y = np.random.uniform(-self.hy, self.hy)

            dist_to_center = np.linalg.norm((x, y))
            dist_to_robot = np.linalg.norm((x, y) - robot_position)

            center_tolerance = target_radius + self.size / 2
            robot_tolerance = robot_size + self.size / 2

            has_valid_location = dist_to_center > center_tolerance and dist_to_robot > robot_tolerance
        
        position = np.array([x, y, self.size / 2])

        self.cube.set_world_pose(position, cube_orientation)