import numpy as np
import math

from omni.physx import get_physx_scene_query_interface

SENSOR_MIN_RANGE = 0.30
SENSOR_MAX_RANGE = 6.0


class UltrasonicSensors:
    def __init__(
            self,
            nb_sensors: int = 4,
            sensor_height: float = 0.2,
            sensor_config:tuple = (180.0, 15.0, 5)
    ):
        """Simulates directional ultrasonic sensors using Isaac Sim PhysX
        raycasting. Rays are cast horizontally from the robot's position
        against all physics collision geometry in the scene — including cube
        obstacles, arena walls, and any other colliders — making this
        implementation correct for both walled and open environments.

        The number of sensors determines which directions are active:
            0: no sensors, get_distances() returns an empty array.
            1: front only.
            2: front and back.
            4: front, back, left, right.

        Sensor directions are defined in the robot's body frame (X forward,
        Y left, Z up) and rotated into the world frame at query time using
        the robot's current yaw.

        The PhysX query interface is not initialised here — call initialize()
        explicitly after world.reset() to ensure the physics simulation is
        ready before the first raycast.

        Args:
            nb_sensors (int): Number of active sensors. Must be 0, 1, 2,
                or 4. Any other value raises a ValueError. Defaults to 4.
            sensor_height (float): Height above the robot's reported z
                position at which rays are cast, in metres. Should
                approximate the physical height of the sensors on the robot.
                Defaults to 0.2.
            sensor_config (tuple): A 3-element tuple
                (lateral_sensors_angle, cone_angle, nb_rays) where:
                - lateral_sensors_angle (float): Angle in degrees between
                  the two lateral sensors. At 180° the sensors are
                  orthogonal to each other and to the front/back sensors.
                  Values below 180° rotate the lateral sensors toward the
                  front; values above 180° rotate them toward the back.
                  Only has an effect when nb_sensors=4. Defaults to 180.0.
                - cone_angle (float): Full cone angle in degrees over which
                  nb_rays are spread for each sensor. Defaults to 15.0.
                - nb_rays (int): Number of rays cast per sensor cone.
                  The minimum distance across all rays is returned as the
                  sensor reading. Defaults to 5.

        Attributes:
            sensor_dirs (dict): Mapping from sensor name to its unit
                direction vector in the robot's body frame. Built from
                nb_sensors at construction time.

        Raises:
            ValueError: If nb_sensors is not one of {0, 1, 2, 4}.
        """
        self.nb_sensors    = nb_sensors
        self.sensor_height = sensor_height
        self.lat_angle           = math.radians(sensor_config[0])
        self.cone_angle = sensor_config[1]
        self.nb_rays = sensor_config[2]

        if self.nb_sensors == 0:
            self.sensor_dirs = {}

        elif self.nb_sensors == 1:
            self.sensor_dirs = {
                "front": np.array([1.0, 0.0, 0.0]),
            }

        elif self.nb_sensors == 2:
            self.sensor_dirs = {
                "front": np.array([ 1.0, 0.0, 0.0]),
                "back":  np.array([-1.0, 0.0, 0.0]),
            }

        elif self.nb_sensors == 4:
            self.sensor_dirs = {
                "front": np.array([ 1.0,  0.0, 0.0]),
                "back":  np.array([-1.0,  0.0, 0.0]),
                "left":  np.array([ math.cos(self.lat_angle/2.0), math.sin(self.lat_angle/2.0), 0.0]),
                "right": np.array([ math.cos(-self.lat_angle/2.0), math.sin(-self.lat_angle/2.0), 0.0]),
            }

        else:
            raise ValueError(
                f"Invalid number of sensors: {self.nb_sensors}. "
                f"Must be one of {{0, 1, 2, 4}}."
            )

        self._ray_prim_path = "/World/SensorRays"

    def initialize(self, render: bool = False) -> None:
        """Initialises the PhysX scene query interface used for raycasting,
        and optionally sets up the USD ray visualisation prim.

        Must be called after world.reset() — the PhysX simulation must be
        fully started before the query interface is valid.

        Args:
            render (bool): If True, creates the USD BasisCurves prim used
                by draw_rays() to visualise sensor rays in the viewport.
                Leave False during headless training. Defaults to False.
        """
        self.query_interface = get_physx_scene_query_interface()

        if render:
            self._init_ray_prim()

    def _init_ray_prim(self) -> None:
        """Creates the USD BasisCurves prim that draw_rays() writes into.
        Called once by initialize() when render=True. If the prim already
        exists in the stage it is reused unchanged.
        """
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()

        if not stage.GetPrimAtPath(self._ray_prim_path):
            curves = UsdGeom.BasisCurves.Define(stage, self._ray_prim_path)
            curves.CreateTypeAttr("linear")
            curves.CreateWidthsAttr([0.01])

    def _distance_to_colour(self, dist: float) -> tuple:
        """Maps a distance in [SENSOR_MIN_RANGE, SENSOR_MAX_RANGE] to an
        RGB colour using a green → yellow → red gradient.

        Close distances (near SENSOR_MIN_RANGE) map to green (0, 1, 0).
        Mid-range distances map to yellow (1, 1, 0).
        Far distances (near SENSOR_MAX_RANGE) map to red (1, 0, 0).

        Args:
            dist (float): Distance in metres.

        Returns:
            tuple[float, float, float]: RGB colour with each component
                in [0.0, 1.0].
        """
        t = (dist - SENSOR_MIN_RANGE) / (SENSOR_MAX_RANGE - SENSOR_MIN_RANGE)
        t = float(np.clip(t, 0.0, 1.0))

        r = t
        g = 1.0 - t
        b = 0.2 * (1.0 - abs(2*t - 1))

        return (r, g, b)

    def draw_rays(
            self,
            origin: np.ndarray,
            yaw: float,
    ) -> None:
        """Draws the sensor cone rays into the Isaac Sim viewport by writing
        line segments into the USD BasisCurves prim created by initialize().

        Uses self.cone_angle and self.nb_rays (set from sensor_config at
        construction time) so the visualisation always matches the raycasting
        behaviour. Each ray is coloured according to the distance it measures:
        green for close obstacles, yellow for mid-range, red for nothing
        nearby. The prim is updated on every call so rays move with the robot.

        Only has an effect when initialize(render=True) was called. Exits
        silently if the prim does not exist.

        Args:
            origin (np.ndarray): World-frame robot position [x, y, z].
            yaw (float): Robot yaw angle in radians.
        """
        import omni.usd
        from pxr import UsdGeom, Gf, Vt

        stage = omni.usd.get_context().get_stage()
        prim  = stage.GetPrimAtPath(self._ray_prim_path)
        if not prim:
            return

        origin_3d = np.array(origin, dtype=float)
        origin_3d[2] += self.sensor_height

        c        = math.cos(yaw)
        s        = math.sin(yaw)
        rot      = np.array([[c, -s, 0],
                              [s,  c, 0],
                              [0,  0, 1]])
        cone_rad = math.radians(self.cone_angle)

        points = []
        counts = []
        colors = []

        for name in self.sensor_dirs:
            world_dir   = rot @ self.sensor_dirs[name]
            world_dir  /= (np.linalg.norm(world_dir) + 1e-8)
            central_yaw = math.atan2(world_dir[1], world_dir[0])

            for offset in np.linspace(-cone_rad / 2, cone_rad / 2, self.nb_rays):
                ray_yaw = central_yaw + offset
                ray_dir = np.array([math.cos(ray_yaw), math.sin(ray_yaw), 0.0])

                dist = self._single_raycast(origin_3d, ray_dir)
                dist = float(np.clip(dist, SENSOR_MIN_RANGE, SENSOR_MAX_RANGE))
                end  = origin_3d + ray_dir * dist

                points.append(Gf.Vec3f(*origin_3d.tolist()))
                points.append(Gf.Vec3f(*end.tolist()))
                counts.append(2)

                colour = self._distance_to_colour(dist)
                colors.append(Gf.Vec3f(*colour))

        curves = UsdGeom.BasisCurves(prim)
        UsdGeom.Gprim(prim).CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform)
        curves.GetPointsAttr().Set(Vt.Vec3fArray(points))
        curves.GetCurveVertexCountsAttr().Set(Vt.IntArray(counts))
        UsdGeom.Gprim(prim).GetDisplayColorAttr().Set(Vt.Vec3fArray(colors))

    def _single_raycast(
            self,
            origin: np.ndarray,
            direction: np.ndarray
    ) -> float:
        """Casts a single ray using the PhysX scene query interface and
        returns the distance to the nearest hit.

        Args:
            origin (np.ndarray): Ray origin in world frame [x, y, z].
            direction (np.ndarray): Unit direction vector [dx, dy, dz].

        Returns:
            float: Distance to the nearest solid collision hit in metres,
                or SENSOR_MAX_RANGE if nothing was hit within range.
        """
        hit = self.query_interface.raycast_closest(
            origin.tolist(),
            direction.tolist(),
            SENSOR_MAX_RANGE,
            False,
        )

        if hit["hit"]:
            return hit["distance"]

        return SENSOR_MAX_RANGE

    def _cone_raycast(
            self,
            origin: np.ndarray,
            central_dir: np.ndarray,
    ) -> float:
        """Approximates an ultrasonic cone sensor by casting self.nb_rays
        spread evenly across a horizontal fan of self.cone_angle degrees
        centred on central_dir, and returning the minimum distance found.

        Rays are kept strictly horizontal (z component = 0) to avoid
        self-collision with the robot's own collision mesh.

        Args:
            origin (np.ndarray): Ray origin [x, y, z] in world frame,
                already offset vertically by sensor_height.
            central_dir (np.ndarray): Unit vector of the cone's central
                axis in world frame.

        Returns:
            float: Minimum distance across all ray hits, clipped to
                [SENSOR_MIN_RANGE, SENSOR_MAX_RANGE].
        """
        min_dist    = SENSOR_MAX_RANGE
        cone_rad    = math.radians(self.cone_angle)
        central_yaw = math.atan2(central_dir[1], central_dir[0])

        for offset in np.linspace(-cone_rad / 2, cone_rad / 2, self.nb_rays):
            yaw     = central_yaw + offset
            ray_dir = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            dist    = self._single_raycast(origin, ray_dir)
            min_dist = min(min_dist, dist)

        return float(np.clip(min_dist, SENSOR_MIN_RANGE, SENSOR_MAX_RANGE))

    def get_distances(
            self,
            origin: np.ndarray,
            yaw: float,
    ) -> np.ndarray:
        """Returns the distance readings for all active sensors at the
        robot's current pose.

        Args:
            origin (np.ndarray): World-frame robot position [x, y, z].
            yaw (float): Robot yaw angle in radians.

        Returns:
            np.ndarray: Array of shape (nb_sensors,) and dtype float32
                containing distances in metres. Returns an empty array
                of shape (0,) when nb_sensors=0.
        """
        distances = []

        origin    = np.array(origin, dtype=float)
        origin[2] += self.sensor_height

        c   = math.cos(yaw)
        s   = math.sin(yaw)
        rot = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1],
        ])

        for name in self.sensor_dirs:
            world_dir  = rot @ self.sensor_dirs[name]
            world_dir /= (np.linalg.norm(world_dir) + 1e-8)

            dist = self._cone_raycast(origin, world_dir)
            dist = float(np.clip(dist, SENSOR_MIN_RANGE, SENSOR_MAX_RANGE))
            distances.append(dist)

        return np.array(distances, dtype=np.float32)