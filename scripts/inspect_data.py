import sys, glob, numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from data.dataset import OfflineSkillDataset
from utils.normalization import Normalizer
from utils.track import Track
import pickle
paths = sorted(glob.glob("data/offline/*.pkl"))
ds = OfflineSkillDataset(paths, horizon=20, stride=2)

lens = np.array([len(e["actions"]) for e in ds.episodes])
crash = np.mean([e["crashed"] for e in ds.episodes])
acts = ds.all_actions()
print(f"episodes: {len(ds.episodes)}   windows: {len(ds)}")
print(f"episode length: mean {lens.mean():.0f}, min {lens.min()}, max {lens.max()}")
print(f"crash rate: {crash:.1%}")
print(f"steer: mean {acts[:,0].mean():.3f}, std {acts[:,0].std():.3f}")
print(f"speed: mean {acts[:,1].mean():.2f}, std {acts[:,1].std():.2f}")

norm = Normalizer().fit(acts)
norm.save("checkpoints/action_normalizer.npz")
ds.action_norm = norm
b = ds[0]
print("sample shapes:", {k: tuple(v.shape) for k, v in b.items()})   # states (21,55) actions (20,2) progress (21,)

dev, ratio = [], []
for p in paths:
    d = pickle.load(open(p, "rb"))
    tr = Track("maps", d["map"])
    half = np.minimum(tr.w_right, tr.w_left)
    for e in d["episodes"]:
        xy = e["poses"][:, :2]
        dist = np.linalg.norm(xy[:, None, :] - tr.xy[None, ::5, :], axis=2).min(1)
        idx = np.linalg.norm(xy[:, None, :] - tr.xy[None, ::5, :], axis=2).argmin(1) * 5
        ratio.append(dist / half[idx])
ratio = np.concatenate(ratio)
print(f"lateral offset / half-width: mean {ratio.mean():.2f}, p90 {np.percentile(ratio,90):.2f}, p99 {np.percentile(ratio,99):.2f}")


fig, ax = plt.subplots(1, 4, figsize=(18, 4))
ax[0].hist(acts[:, 0], 60); ax[0].set_title("steer")
ax[1].hist(acts[:, 1], 60); ax[1].set_title("speed")
ax[2].hist(lens, 30); ax[2].set_title("episode length")
for e in ds.episodes[:20]:
    ax[3].plot(e["poses"][:, 0], e["poses"][:, 1], lw=0.7)
ax[3].set_aspect("equal"); ax[3].set_title("trajectories")
plt.tight_layout(); plt.savefig("outputs/data_check.png", dpi=120); plt.show()