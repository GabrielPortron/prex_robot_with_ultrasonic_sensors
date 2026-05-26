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
            perimeter=(2.0, 2.0)
    ):
        
        self.hx = perimeter[0] / 2.0
        self.hy = perimeter[1] / 2.0
    
    def get_distances(self, position, yaw):
        return self.analytic_cast(position[:2], yaw)
    
    def analytic_cast(self, pos2d, yaw):

        x, y = pos2d
        distances = np.zeros(4, dtype=np.float32)

        for i, (_, local_yaw) in enumerate(SENSOR_YAWS.items()):
            world_yaw = yaw + local_yaw
            dx = math.cos(world_yaw)
            dy = math.sin(world_yaw)

            dist = self.ray_vs_aabb(x, y, dx, dy)
            distances[i] = np.clip(dist, SENSOR_MIN_RANGE, SENSOR_MAX_RANGE)
        
        return distances
    
    def ray_vs_aabb(self, ox, oy, dx, dy):

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