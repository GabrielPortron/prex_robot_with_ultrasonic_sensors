from isaacsim.core.api.objects import FixedCuboid

import numpy as np

class Cube:
    def __init__(
            self,
            world,
            scale=(0.2, 0.2, 0.2),
            perimeter=(2.0, 2.0)
    ):
        
        self.world = world
        self.scale = scale
        self.size = self.scale[2]

        self.perimeter = perimeter
        self.length, self.width = self.perimeter

        self.hx = (self.length / 2.0) - self.size / 2.0
        self.hy = (self.width / 2.0) - self.size / 2.0
    
    def create_cube(self):

        prim_path = "/World/Square_Arena/cube"

        self.cube = FixedCuboid(
                prim_path=prim_path,
                name="cube",
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                scale=self.scale
            )

        self.world.scene.add(self.cube)

    def teleport_cube(self, orientation, target_radius, robot_position, robot_size):

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

        self.cube.set_world_pose(position, orientation)

        return position[:2]