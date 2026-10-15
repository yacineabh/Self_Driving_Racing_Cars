import numpy as np

WHEELBASE = 0.3302      # f1tenth gym default
MAX_STEER = 0.4189      # rad

class PurePursuit:
    def __init__(self, track, speed=3.0, lookahead_base=0.8, lookahead_gain=0.25):
        self.track, self.speed = track, speed
        self.lb, self.lg = lookahead_base, lookahead_gain

    def act(self, x, y, theta, v):
        # theta → car's orientation
        L = self.lb + self.lg * max(v, 0.0)          # longer lookahead when faster
        gx, gy = self.track.lookahead_point(x, y, L)
        dx, dy = gx - x, gy - y
        local_y = -np.sin(theta) * dx + np.cos(theta) * dy   # goal's Convert the target into the car's coordinate system following the car's orientation
        # local_y > 0  → target is on the LEFT
        # local_y < 0  → target is on the RIGHT
        # local_y ≈ 0  → target is straight ahead
        steer = np.arctan2(2 * WHEELBASE * local_y, L ** 2)  # curvature = 2*y / L^2
        # if local_y = large then steer = large, if local_y ≈ 0 then steer ≈ 0
        # If the target is farther away, L becomes larger .Therefore the steering becomes smaller.

        return float(np.clip(steer, -MAX_STEER, MAX_STEER)), self.speed