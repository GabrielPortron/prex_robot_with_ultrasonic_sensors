from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from isaacsim.core.api import World

from arena import Arena
from robot import Create3Robot

class PrexIsaacEnv:
    def __init__(
            self,
            arena_geometry=[(2.0, 2.0), 0.2, 0.5],
    ):
        
        self.perimeter = arena_geometry[0]
        self.depth = arena_geometry[1]
        self.heigth = arena_geometry[2]
    
    def launch(self):

        # --- 1. Create world --------------------------------------------
        self.world = World()

        # --- 2. Build Arena ---------------------------------------------
        self.arena = Arena(
            world=self.world,
            perimeter=self.perimeter,
            depth=self.depth,
            height=self.heigth
        )
        self.arena.build()

        # --- 3. Add Robot -----------------------------------------------
        self.robot = Create3Robot(world=self.world)
        self.robot.load()

        # --- ?. Physic Initialization -----------------------------------
        self.world.reset()

        # --- After First Reset ! ----------------------------------------
        self.robot.initialize()
