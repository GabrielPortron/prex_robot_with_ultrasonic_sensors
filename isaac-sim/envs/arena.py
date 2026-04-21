from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from isaacsim.core.api.objects import FixedCuboid

import numpy as np

class Arena:
    def __init__(
            self,
            world,
            perimeter=(2.0, 2.0),
            depth=0.2,
            height=0.5
    ):
        
        self.world = world

        self.perimeter = perimeter
        self.depth = depth
        self.height = height

        self.corner_size = self.depth
        self.length, self.width = self.perimeter
        self.offset_length = (self.width + self.depth) / 2
        self.offset_width = (self.length + self.depth) / 2
    
    def create_wall(self, name, position, scale):

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