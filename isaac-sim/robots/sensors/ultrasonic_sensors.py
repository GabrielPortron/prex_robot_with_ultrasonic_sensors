import numpy as np
import math

from omni.physx import get_physx_scene_query_interface

from utils.utils import euler_to_quaternion

SENSOR_MIN_RANGE = 0.30
SENSOR_MAX_RANGE = 20.0


class UltrasonicSensors:

    def __init__(
        self,
        nb_sensors: int = 4,
        sensor_height: float = 0.2,
    ):
        self.nb_sensors = nb_sensors
        self.sensor_height = sensor_height

        self.sensors = []

        if self.nb_sensors == 0:

            self.sensor_dirs = {}

        elif self.nb_sensors == 1:

            self.sensor_dirs = {
                "front": np.array([1.0, 0.0, 0.0]),
            }

        elif self.nb_sensors == 2:

            self.sensor_dirs = {
                "front": np.array([1.0, 0.0, 0.0]),
                "back": np.array([-1.0, 0.0, 0.0]),
            }

        elif self.nb_sensors == 4:

            self.sensor_dirs = {
                "front": np.array([1.0, 0.0, 0.0]),
                "back": np.array([-1.0, 0.0, 0.0]),
                "left": np.array([0.0, 1.0, 0.0]),
                "right": np.array([0.0, -1.0, 0.0]),
            }

        else:

            raise ValueError(
                f"Invalid number of sensors: {self.nb_sensors}"
            )

        self.query_interface = get_physx_scene_query_interface()
        # self.query_interface.set_exclude_self_collision(False)

    def _raycast(self, origin, direction):
        """
        PhysX raycast.
        """
        hit = self.query_interface.raycast_closest(
            origin.tolist(),
            direction.tolist(),
            SENSOR_MAX_RANGE,
            False
        )

        if hit["hit"]:
            return hit["distance"]

        return SENSOR_MAX_RANGE

    def get_distances(self, origin, yaw):

        distances = []

        origin = np.array(origin, dtype=float)
        origin[2] += self.sensor_height

        for name in self.sensor_dirs:

            # 2. rotate by yaw around Z axis
            c = math.cos(yaw)
            s = math.sin(yaw)

            rot_matrix = np.array([
                [c, -s, 0],
                [s,  c, 0],
                [0,  0, 1]
            ])

            world_dir = rot_matrix @ self.sensor_dirs[name]

            # 3. normalize
            world_dir = world_dir / (np.linalg.norm(world_dir) + 1e-8)

            # 4. raycast
            dist = self._raycast(origin, world_dir)
            # 5. clamp
            dist = np.clip(dist, SENSOR_MIN_RANGE, SENSOR_MAX_RANGE)

            distances.append(dist)

        return np.array(distances, dtype=np.float32)
