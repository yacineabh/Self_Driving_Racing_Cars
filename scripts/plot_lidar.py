import sys, yaml, gym, numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from utils.track import Track
from utils.pure_pursuit import PurePursuit
from utils.lidar import LidarProcessor

def load_map(name):
    cfg = yaml.safe_load(open(f"maps/{name}/{name}_map.yaml"))
    img = plt.imread(f"maps/{name}/{name}_map.png")
    if img.ndim == 3:
        img = img[..., 0]
    ox, oy = cfg["origin"][:2]
    res = cfg["resolution"]
    h, w = img.shape
    return img, [ox, ox + w * res, oy, oy + h * res]

name = sys.argv[1] if len(sys.argv) > 1 else "Spielberg"
SNAP_TICK = 150

track = Track("maps", name)
lidar = LidarProcessor()
env = gym.make("f110_gym:f110-v0", map=f"maps/{name}/{name}_map",
               map_ext=".png", num_agents=1)
obs, _, done, _ = env.reset(track.start_pose())
pp = PurePursuit(track, speed=3.0)

xs, ys, prog, states, snap = [], [], [], [], None
for t in range(600):
    x, y, th = obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]
    v = obs["linear_vels_x"][0]
    steer, spd = pp.act(x, y, th, v)
    for _ in range(5):
        obs, _, done, _ = env.step(np.array([[steer, spd]]))
    x, y, th = obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]
    v = obs["linear_vels_x"][0]
    xs.append(x); ys.append(y); prog.append(track.progress(x, y))
    states.append(lidar.state(obs["scans"][0], v))
    if t == SNAP_TICK:
        snap = (x, y, th, np.array(obs["scans"][0]).copy())
    if obs["collisions"][0]:
        print("collision at tick", t); break

states = np.array(states)
print("state array:", states.shape, "min", states.min(), "max", states.max())

img, extent = load_map(name)
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].imshow(img, cmap="gray", extent=extent, origin="upper")
ax[0].plot(xs, ys, "b-", lw=1, label="trajectory")
if snap:
    sx, sy, sth, scan = snap
    r = lidar.beams(scan) * lidar.max_range
    a = sth + lidar.angles                       # beam angles in world frame
    ex, ey = sx + r * np.cos(a), sy + r * np.sin(a)
    for i in range(len(r)):
        ax[0].plot([sx, ex[i]], [sy, ey[i]], "r-", lw=0.4)
    ax[0].plot(ex, ey, "r.", ms=3)
    ax[0].plot(sx, sy, "go", ms=7)
ax[0].set_title(f"{name}: trajectory + LiDAR snapshot (tick {SNAP_TICK})")
ax[0].set_aspect("equal")

ax[1].plot(prog); ax[1].set_title("progress along centerline (m)"); ax[1].set_xlabel("control tick")

ax[2].imshow(states.T, aspect="auto", cmap="viridis")
ax[2].set_title("state over time (rows = 54 beams + speed)"); ax[2].set_xlabel("control tick")

plt.tight_layout()
plt.savefig(f"outputs/lidar_check_{name}.png", dpi=120)
plt.show()