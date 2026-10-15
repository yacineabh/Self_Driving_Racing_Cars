import numpy as np

class LidarProcessor:
    """Raw 1080-beam scan -> compact normalized state vector."""

    def __init__(self, n_raw=1080, raw_fov=4.7, use_fov_deg=180.0,
                 n_beams=54, max_range=10.0, v_max=8.0):
        self.max_range, self.v_max = max_range, v_max
        raw_angles = np.linspace(-raw_fov / 2, raw_fov / 2, n_raw)   # radians, beam 0 = right side
        half = np.deg2rad(use_fov_deg) / 2
        self.angles = np.linspace(-half, half, n_beams)               # angles we keep (car frame)
        self.idx = np.array([np.argmin(np.abs(raw_angles - a)) for a in self.angles])
        self.n_beams = n_beams
        self.state_dim = n_beams + 1                                  # beams + speed

    def beams(self, scan):
        # Normalize the selected beams and replacing NaN/inf with max_range + clip to [0, max_range]
        r = np.asarray(scan, dtype=np.float32)[self.idx]
        r = np.nan_to_num(r, nan=self.max_range, posinf=self.max_range)
        return np.clip(r, 0.0, self.max_range) / self.max_range

    def state(self, scan, v):
        # 54 beams + 1 speed = 55-dim state vector
        s = np.concatenate([self.beams(scan), [np.clip(v / self.v_max, 0.0, 1.0)]])
        return s.astype(np.float32)