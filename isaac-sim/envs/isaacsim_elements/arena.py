from isaacsim.core.api.objects import FixedCuboid

import numpy as np

class Arena:
    def __init__(
            self,
            world,
            perimeter: tuple[float, float] =(2.0, 2.0),
            depth: float =0.2,
            height: float =0.5
    ):
        """The class that creates a rectangular arena in IsaacSim.

        Args:
            world : The world in which the arena is created.
            perimeter (tuple[float, float], optional): The dimensions of the arena: length and width. Defaults to (2.0, 2.0).
            depth (float, optional): The depth of the walls of the arena. Defaults to 0.2.
            height (float, optional): The height of the walls of the arena. Defaults to 0.5.
        """
        
        self.world = world

        self.perimeter = perimeter
        self.depth = depth
        self.height = height

        self.corner_size = self.depth
        self.length, self.width = self.perimeter
        self.offset_length = (self.width + self.depth) / 2
        self.offset_width = (self.length + self.depth) / 2
    
    def create_wall(self, name, position, scale):
        """The function that creates a wall to a given posiiton and scale.

        Args:
            name (str): The name of the wall.
            position (tuple[float, float, float]): The position of the wall.
            scale (tuple[float, float, float]): The scale of the wall.
        """

        prim_path = f"/World/Square_Arena/{name}"

        self.world.scene.add(
            FixedCuboid(
                prim_path=prim_path,
                name=name,
                position=position,
                scale=scale
            )
        )

    def build(self):
        """The function that builds the entire arena and sets the ground in the world.
        """
        
        self.world.scene.add_default_ground_plane()

        self.create_wall(
            name="North",
            position=np.array([0.0, self.offset_length, self.height / 2]),
            scale=np.array([self.length, self.depth, self.height])
        )
        self.create_wall(
            name="South",
            position=np.array([0.0, -self.offset_length, self.height / 2]),
            scale=np.array([self.length, self.depth, self.height])
        )
        self.create_wall(
            name="East",
            position=np.array([self.offset_width, 0.0, self.height / 2]),
            scale=np.array([self.depth, self.width, self.height])
        )
        self.create_wall(
            name="West",
            position=np.array([-self.offset_width, 0.0, self.height / 2]),
            scale=np.array([self.depth, self.width, self.height])
        )

        self.create_wall(
            name="Corner1",
            position=np.array([self.offset_width, self.offset_length, self.height / 2]),
            scale=np.array([self.corner_size, self.corner_size, self.height])
        )
        self.create_wall(
            name="Corner2",
            position=np.array([self.offset_width, -self.offset_length, self.height / 2]),
            scale=np.array([self.corner_size, self.corner_size, self.height])
        )
        self.create_wall(
            name="Corner3",
            position=np.array([-self.offset_width, -self.offset_length, self.height / 2]),
            scale=np.array([self.corner_size, self.corner_size, self.height])
        )
        self.create_wall(
            name="Corner4",
            position=np.array([-self.offset_width, self.offset_length, self.height / 2]),
            scale=np.array([self.corner_size, self.corner_size, self.height])
        )