import numpy as np
import math

SENSOR_YAWS = {
    "front": 0.0,
    "left": math.pi / 2.0,
    "back": math.pi,
    "right": -math.pi / 2.0
}

SENSOR_MIN_RANGE = 0.30
SENSOR_MAX_RANGE = 100.0

class UltrasonicSensors:
    def __init__(
            self,
            perimeter: tuple[float, float] = (2.0, 2.0)
    ):
        """Simulates four ultrasonic sensors (front, left, back, right) mounted
        on a robot navigating a rectangular arena. Distances are computed
        analytically using ray-AABB intersection rather than physics raycasts,
        making them fast and noise-free for RL training.

        Args:
            perimeter (tuple[float, float], optional): Width and height of the
                arena in metres. The arena is assumed to be centred at the
                origin, so walls are placed at ±perimeter[0]/2 on the X axis
                and ±perimeter[1]/2 on the Y axis. Defaults to (2.0, 2.0).

        Attributes:
            hx (float): Half-width of the arena (X axis).
            hy (float): Half-height of the arena (Y axis).
        """
        self.hx = perimeter[0] / 2.0
        self.hy = perimeter[1] / 2.0

    def get_distances(
            self,
            position: np.ndarray,
            yaw: float
    ) -> np.ndarray:
        """Returns the distance readings of the four sensors given the robot's
        current pose. This is the main public interface of the class.

        Args:
            position (np.ndarray): World-frame position of the robot as a 1D
                array of at least 3 elements [x, y, z]. Only the x and y
                components are used.
            yaw (float): Yaw angle of the robot in radians, measured
                counter-clockwise from the positive X axis.

        Returns:
            np.ndarray: Array of shape (4,) and dtype float32 containing the
                distances [front, left, back, right] in metres, clipped to
                [SENSOR_MIN_RANGE, SENSOR_MAX_RANGE].
        """
        return self.analytic_cast(position[:2], yaw)

    def analytic_cast(
            self,
            pos2d: np.ndarray,
            yaw: float
    ) -> np.ndarray:
        """Casts one ray per sensor direction and collects the distances to the
        nearest arena wall. Sensor directions are defined in the robot's body
        frame (SENSOR_YAWS) and rotated into the world frame using the robot's
        yaw before casting.

        Args:
            pos2d (np.ndarray): 2D position of the robot [x, y] in metres.
            yaw (float): Yaw angle of the robot in radians.

        Returns:
            np.ndarray: Array of shape (4,) and dtype float32 with distances
                [front, left, back, right], clipped to
                [SENSOR_MIN_RANGE, SENSOR_MAX_RANGE].
        """
        x, y = pos2d
        distances = np.zeros(4, dtype=np.float32)

        for i, (_, local_yaw) in enumerate(SENSOR_YAWS.items()):
            world_yaw = yaw + local_yaw
            dx = math.cos(world_yaw)
            dy = math.sin(world_yaw)

            dist = self.ray_vs_aabb(x, y, dx, dy)
            distances[i] = np.clip(dist, SENSOR_MIN_RANGE, SENSOR_MAX_RANGE)

        return distances

    def ray_vs_aabb(
            self,
            ox: float,
            oy: float,
            dx: float,
            dy: float
    ) -> float:
        """Computes the distance from a ray origin to the nearest wall of the
        axis-aligned bounding box (AABB) that defines the arena.

        The ray is defined by its origin (ox, oy) and a unit direction (dx, dy).
        For each pair of parallel walls the function computes the ray parameter
        t at which the ray crosses the wall plane, then checks whether the
        crossing point lies within the arena bounds on the perpendicular axis.
        The smallest positive valid t is returned.

        Args:
            ox (float): X coordinate of the ray origin in metres.
            oy (float): Y coordinate of the ray origin in metres.
            dx (float): X component of the unit ray direction vector.
            dy (float): Y component of the unit ray direction vector.

        Returns:
            float: Distance to the nearest wall hit in metres. Returns
                SENSOR_MAX_RANGE if no valid intersection is found (e.g. the
                ray is parallel to both wall pairs, which should not occur in
                normal operation).
        """
        t_min = float("inf")

        if abs(dx) > 1e-9:
            for wall_x in (-self.hx, self.hx):
                t = (wall_x - ox) / dx
                if t > 0:
                    hit_y = oy + t * dy
                    if -self.hy <= hit_y <= self.hy:
                        t_min = min(t_min, t)

        if abs(dy) > 1e-9:
            for wall_y in (-self.hy, self.hy):
                t = (wall_y - oy) / dy
                if t > 0:
                    hit_x = ox + t * dx
                    if -self.hx <= hit_x <= self.hx:
                        t_min = min(t_min, t)

        if t_min == float("inf"):
            t_min = SENSOR_MAX_RANGE

        return t_min