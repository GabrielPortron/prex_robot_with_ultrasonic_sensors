from isaacsim.core.api.objects import FixedCuboid

import numpy as np
import math

class Cube:
    def __init__(
            self,
            world,
            dimension: float = 0.3,
            perimeter: tuple[float, float] =(2.0, 2.0)
    ):
        """The class that manages a configurable number of cubic obstacles
        for the robot's training environment.

        Cubes are created at the origin by create_cubes() and repositioned
        at the start of each episode by set_up_all_cubes(). Each cube is
        placed at a random location that avoids the goal area, the robot,
        and all previously placed cubes.

        Args:
            world: The Isaac Sim World instance the cubes will be added to.
            nb_cube (int): Number of cube obstacles to create. If 0, no cubes
                are added to the scene but the manager object is still valid.
            scale (tuple[float, float, float]): Dimensions [x, y, z] of each
                cube in metres. All cubes share the same scale.
                Defaults to (0.2, 0.2, 0.2).
            perimeter (tuple[float, float]): Width and height of the spawn
                region in metres. Cubes are placed within this region with a
                half-size margin from the boundary.
                Defaults to (2.0, 2.0).

        Attributes:
            size (float): The z-dimension of the cube scale, used as the
                collision radius for placement validation.
            hx (float): Maximum x-coordinate for cube spawning, equal to
                half the arena width minus half the cube size.
            hy (float): Maximum y-coordinate for cube spawning, equal to
                half the arena height minus half the cube size.
            cubes (list): List of FixedCuboid instances created by
                create_cubes().
        """
        self.world = world
        self.size = dimension

        self.perimeter = perimeter
        self.length, self.width = self.perimeter

        self.hx = (self.length / 2.0) - self.size / 2.0
        self.hy = (self.width / 2.0) - self.size / 2.0

        self.cubes = []
    
    def create_cubes(self, nb_cubes: int = 0) -> None:
        """Creates nb_cube FixedCuboid instances and adds them to the world
        scene. All cubes are initially placed at the origin — call
        set_up_all_cubes() at the start of each episode to distribute them
        to valid random positions.

        Each cube is given a unique prim path and name based on its index
        (cube0, cube1, ...) and stored in self.cubes for later access by
        teleport_cube() and set_up_all_cubes().
        """
        for i in range(nb_cubes):

            name = f"cube{i}"

            prim_path = f"/World/Square_Arena/{name}"

            cube = FixedCuboid(
                    prim_path=prim_path,
                    name=name,
                    position=np.array([0.0, 0.0, 0.0]),
                    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                    scale=np.array([self.size, self.size, self.size])
                )

            self.world.scene.add(cube)
            self.cubes.append(cube)

    def teleport_cube(
            self,
            cube,
            target_radius: float,
            obstacle_positions: list,
            robot_size: float
    ) -> np.ndarray:
        """Repositions a single cube to a random valid location that avoids
        all known obstacles.

        A location is valid if it is sufficiently far from:
            - The goal (origin): distance > target_radius + cube_size / 2
            - The robot (first entry in obstacle_positions):
                distance > robot_size + cube_size / 2
            - All previously placed cubes (remaining entries):
                distance > cube_size * 3 / 2

        The function samples uniformly until a valid location is found.
        A random yaw is also assigned so cubes are not all axis-aligned.

        Args:
            cube: The FixedCuboid instance to reposition.
            target_radius (float): Radius of the goal exclusion zone in
                metres.
            obstacle_positions (list[np.ndarray]): List of 2D positions
                [x, y] of obstacles already placed. The first entry is
                always the robot position; subsequent entries are previously
                placed cubes.
            robot_size (float): Collision radius of the robot in metres,
                used to compute the minimum distance to the robot.

        Returns:
            np.ndarray: The 2D position [x, y] of the placed cube, to be
                appended to obstacle_positions before placing the next cube.
        """
        cube_yaw = np.random.uniform(-math.pi, math.pi)
        cube_orientation = np.array([math.cos(cube_yaw/2.0), 0.0, 0.0, math.sin(cube_yaw/2.0)])
        
        valid_locations = [False for _ in range(len(obstacle_positions) + 1)]
        has_valid_location = False

        while not has_valid_location:
            x = np.random.uniform(-self.hx, self.hx)
            y = np.random.uniform(-self.hy, self.hy)

            dist_to_obstacle = [np.linalg.norm((x, y))]
            obstacle_range = [target_radius + self.size/2]
            
            for i in range(len(obstacle_positions)):
                dist_to_obstacle.append(np.linalg.norm((x, y) - obstacle_positions[i]))
                if i == 0:
                    obstacle_range.append(robot_size + self.size / 2)
                else:
                    obstacle_range.append((self.size * 3) / 2)

            nb_valid = 0
            for i in range(len(valid_locations)):
                if dist_to_obstacle[i] > obstacle_range[i]:
                    valid_locations[i] = True
                    nb_valid += 1
            
            if nb_valid == len(valid_locations):
                has_valid_location = True
        
        cube_position = np.array([x, y, self.size / 2])

        cube.set_world_pose(cube_position, cube_orientation)

        return cube_position[:2]
    
    def set_up_all_cubes(
            self,
            target_radius: float,
            robot_position: np.ndarray,
            robot_size: float,
            nb_cubes: int = 0,
    ) -> None:
        """Repositions all nb_cube cubes at the start of an episode by
        calling teleport_cube() sequentially. Each cube's placed position
        is added to the obstacle list before placing the next one, so
        cubes are guaranteed not to overlap each other or the robot.

        Args:
            target_radius (float): Radius of the goal exclusion zone in
                metres, passed to teleport_cube().
            robot_position (np.ndarray): Current 2D robot position [x, y]
                in metres, used as the first entry in obstacle_positions.
            robot_size (float): Collision radius of the robot in metres,
                passed to teleport_cube().
        """
        positions = [robot_position]

        for i in range(nb_cubes):
            cube_position = self.teleport_cube(self.cubes[i], target_radius, positions, robot_size)
            positions.append(cube_position)
    
