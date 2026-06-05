from isaacsim.core.api.objects import FixedCuboid
import numpy as np


class Arena:
    def __init__(
            self,
            world,
            perimeter: tuple[float, float] = (2.0, 2.0),
            depth: float = 0.2,
            height: float = 0.5
    ):
        """Builds a rectangular walled arena in Isaac Sim using FixedCuboid
        objects. The arena consists of four axis-aligned walls (North, South,
        East, West) and four corner fill pieces that close the gaps where
        walls meet.

        The arena is centred at the world origin. Wall positions are computed
        automatically from the perimeter and depth so that the inner usable
        area is exactly perimeter[0] x perimeter[1] metres.

        Args:
            world: The Isaac Sim World instance the arena will be added to.
            perimeter (tuple[float, float]): Inner dimensions of the arena
                [length, width] in metres. length corresponds to the X axis
                and width to the Y axis. Defaults to (2.0, 2.0).
            depth (float): Thickness of each wall in metres. Also used as the
                side length of the corner fill pieces. Defaults to 0.2.
            height (float): Height of the walls in metres. All walls and
                corners share the same height. Defaults to 0.5.

        Attributes:
            corner_size (float): Side length of the corner fill pieces, equal
                to depth.
            length (float): Inner arena length along the X axis in metres.
            width (float): Inner arena width along the Y axis in metres.
            offset_length (float): Distance from the origin to the centre of
                the North/South walls along the Y axis, equal to
                (width + depth) / 2.
            offset_width (float): Distance from the origin to the centre of
                the East/West walls along the X axis, equal to
                (length + depth) / 2.
        """
        self.world = world

        self.perimeter = perimeter
        self.depth     = depth
        self.height    = height

        self.corner_size   = self.depth
        self.length, self.width = self.perimeter
        self.offset_length = (self.width  + self.depth) / 2
        self.offset_width  = (self.length + self.depth) / 2

    def create_wall(
            self,
            name: str,
            position: np.ndarray,
            scale: np.ndarray
    ) -> None:
        """Creates a single FixedCuboid wall and adds it to the world scene.

        All walls share the prim path prefix /World/Square_Arena/. The
        FixedCuboid has no physics dynamics — it is a static collision object
        that the robot cannot push.

        Args:
            name (str): Unique name for this wall, used both as the USD prim
                name and as the scene object name. Examples: "North", "Corner1".
            position (np.ndarray): World-frame centre position [x, y, z] of
                the wall in metres. The z component should be height / 2 so
                the wall sits on the ground plane.
            scale (np.ndarray): Dimensions [x, y, z] of the wall in metres,
                corresponding to its extent along each axis.
        """
        prim_path = f"/World/Square_Arena/{name}"

        self.world.scene.add(
            FixedCuboid(
                prim_path = prim_path,
                name      = name,
                position  = position,
                scale     = scale,
            )
        )

    def build(self) -> None:
        """Builds the complete arena by creating all four walls and four
        corner pieces via create_wall().

        Wall layout (viewed from above, X right, Y up):

                Corner4   North    Corner1
                  ┌─────────────────┐
            West  │                 │  East
                  └─────────────────┘
                Corner3   South    Corner2

        The four main walls span the full inner dimension of the arena on
        their long axis. The four corner pieces fill the square gaps at each
        intersection so there are no holes in the enclosure.

        Does not add a ground plane — call world.scene.add_default_ground_plane()
        separately before or after build() if a ground is needed.
        """
        self.create_wall(
            name     = "North",
            position = np.array([0.0, self.offset_length, self.height / 2]),
            scale    = np.array([self.length, self.depth, self.height])
        )
        self.create_wall(
            name     = "South",
            position = np.array([0.0, -self.offset_length, self.height / 2]),
            scale    = np.array([self.length, self.depth, self.height])
        )
        self.create_wall(
            name     = "East",
            position = np.array([self.offset_width, 0.0, self.height / 2]),
            scale    = np.array([self.depth, self.width, self.height])
        )
        self.create_wall(
            name     = "West",
            position = np.array([-self.offset_width, 0.0, self.height / 2]),
            scale    = np.array([self.depth, self.width, self.height])
        )
        self.create_wall(
            name     = "Corner1",
            position = np.array([ self.offset_width,  self.offset_length, self.height / 2]),
            scale    = np.array([self.corner_size, self.corner_size, self.height])
        )
        self.create_wall(
            name     = "Corner2",
            position = np.array([ self.offset_width, -self.offset_length, self.height / 2]),
            scale    = np.array([self.corner_size, self.corner_size, self.height])
        )
        self.create_wall(
            name     = "Corner3",
            position = np.array([-self.offset_width, -self.offset_length, self.height / 2]),
            scale    = np.array([self.corner_size, self.corner_size, self.height])
        )
        self.create_wall(
            name     = "Corner4",
            position = np.array([-self.offset_width,  self.offset_length, self.height / 2]),
            scale    = np.array([self.corner_size, self.corner_size, self.height])
        )