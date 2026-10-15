import numpy as np

class Track:
    def __init__(self, map_dir, name):
        d = np.loadtxt(f"{map_dir}/{name}/{name}_centerline.csv",
                       delimiter=",", skiprows=1)
        self.xy = d[:, :2]                       # centerline points (meters)
        self.w_right, self.w_left = d[:, 2], d[:, 3]   # track widths, useful later
        self.n = len(self.xy)
        seg = np.linalg.norm(np.diff(self.xy, axis=0, append=self.xy[:1]), axis=1)
        self.s = np.concatenate([[0], np.cumsum(seg)[:-1]])  # arclength at each point
        self.length = seg.sum()

    def start_pose(self):
        d = self.xy[1] - self.xy[0]
        return np.array([[*self.xy[0], np.arctan2(d[1], d[0])]])

    def nearest(self, x, y):
        return int(np.argmin(np.sum((self.xy - [x, y]) ** 2, axis=1)))

    def progress(self, x, y):
        """Arclength position along the centerline (meters)."""
        return self.s[self.nearest(x, y)]

    def lookahead_point(self, x, y, L):
        i = self.nearest(x, y)
        for k in range(self.n):
            j = (i + k) % self.n
            if np.hypot(*(self.xy[j] - [x, y])) >= L:
                return self.xy[j]
        return self.xy[i]