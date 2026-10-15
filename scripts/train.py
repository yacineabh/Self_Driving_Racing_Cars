"""train.py  -  Train the skill model (NaiveVI or CausalEM).

Usage
-----
  python scripts/train.py                                     # naive VI, 20 epochs
  python scripts/train.py --trainer causal_em --epochs 20
  python scripts/train.py --data-glob "data/offline/*_v2*.pkl"
"""
import sys, glob, os, time, argparse, yaml
from collections import defaultdict
import numpy as np, torch
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from torch.utils.data import DataLoader
from data.dataset import OfflineSkillDataset
from utils.normalization import Normalizer
from models.skill_model import SkillModel
from training.trainer import NaiveVITrainer, CausalEMTrainer

# -- argument parsing ----------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--config",     default="configs/default.yaml")
ap.add_argument("--epochs",     type=int, default=None)
ap.add_argument("--trainer",    choices=["naive_vi", "causal_em"], default="naive_vi")
ap.add_argument("--data-dir",   default=None, help="override collection.out_dir")
ap.add_argument("--data-glob",  default=None,
                help="explicit glob pattern for pkl files, e.g. 'data/offline/*_v2*.pkl'")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
if args.data_dir:
    cfg["collection"]["out_dir"] = args.data_dir

tc, mc = cfg["training"], cfg["model"]
T      = cfg["dataset"]["horizon"]
epochs = args.epochs or tc["epochs"]

torch.manual_seed(cfg["seed"])
np.random.seed(cfg["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# -- dataset -------------------------------------------------------------------
if args.data_glob:
    all_paths = sorted(glob.glob(args.data_glob))
else:
    data_dir  = cfg["collection"]["out_dir"]
    all_paths = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))

if not all_paths:
    sys.exit(f"No .pkl files found.  data-glob={args.data_glob!r}")

print(f"Loading {len(all_paths)} pickle file(s):")
for p in all_paths:
    print(f"  {p}")

full = OfflineSkillDataset(all_paths, horizon=T, stride=cfg["dataset"]["stride"])
train_ds, val_ds = full.split(tc["val_frac"], seed=cfg["seed"])

# Fit normalizer on training actions ONLY
norm = Normalizer().fit(train_ds.all_actions())
train_ds.action_norm = norm
val_ds.action_norm   = norm
os.makedirs(tc["checkpoint_dir"], exist_ok=True)
norm.save(os.path.join(tc["checkpoint_dir"], "action_normalizer.npz"))

print(f"train windows: {len(train_ds)}   val windows: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=tc["batch_size"],
                          shuffle=True,  drop_last=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=512,
                          shuffle=False, num_workers=0)

# -- model ---------------------------------------------------------------------
model = SkillModel(**mc).to(device)
TrainerCls = CausalEMTrainer if args.trainer == "causal_em" else NaiveVITrainer
trainer    = TrainerCls(model, tc)

# -- evaluation ----------------------------------------------------------------
@torch.no_grad()
def evaluate(max_batches=40):
    model.eval()
    agg = defaultdict(list)
    for i, b in enumerate(val_loader):
        if i >= max_batches:
            break
        b = {k: v.to(device) for k, v in b.items()}
        for k, v in model(b["states"], b["actions"], b["progress"]).items():
            agg[k].append(v.item())
        for k, v in model.diagnostics(b["states"], b["actions"],
                                       b["progress"]).items():
            agg[k].append(v)
    model.train()
    return {k: float(np.mean(v)) for k, v in agg.items()}

# -- checkpointing -------------------------------------------------------------
stem = f"oposm_{args.trainer}"

def save(path, ep):
    torch.save({"model": model.state_dict(), "cfg": cfg,
                "epoch": ep, "hist": hist, "trainer": args.trainer}, path)

hist, best = [], float("inf")

# -- training loop -------------------------------------------------------------
for ep in range(1, epochs + 1):
    t0  = time.time()
    agg = defaultdict(list)
    for b in train_loader:
        b = {k: v.to(device) for k, v in b.items()}
        for k, v in trainer.step(b).items():
            agg[k].append(v)
    tr = {k: float(np.mean(v)) for k, v in agg.items()}
    va = evaluate()
    hist.append({"epoch": ep, "train": tr, "val": va})

    elapsed = time.time() - t0
    print(f"\nepoch {ep}/{epochs}  ({elapsed:.0f}s)  beta={model.kl_beta:.2f}")
    print(f"  train | policy_nll {tr['policy_nll']:7.2f} | kl {tr['kl']:6.2f}"
          f" | tawm_state {tr['tawm_state']:.3f} | tawm_range {tr.get('tawm_range', 0):.4f}")
    print(f"  val   | policy_nll {va['policy_nll']:7.2f} | kl {va['kl']:6.2f}"
          f" | tawm_state {va['tawm_state']:.3f} | tawm_range {va.get('tawm_range', 0):.4f}")
    print(f"  z-use | tawm: baseline {va['tawm_baseline']:.3f}"
          f"  post {va['tawm_post']:.3f}"
          f"  prior {va['tawm_prior']:.3f}"
          f"  shuffled {va['tawm_shuf']:.3f}")
    print(f"        | policy nll: post {va['pol_post']:.2f}"
          f"  prior {va['pol_prior']:.2f}"
          f"  shuffled {va['pol_shuf']:.2f}")
    print(f"        | range_err_post {va.get('range_err_post', float('nan')):.4f}")

    # Always overwrite final checkpoint
    save(f"{tc['checkpoint_dir']}/{stem}_final.pt", ep)
    # Best checkpoint by total val loss (post warm-up)
    if trainer.step_i >= trainer.warmup and va["total"] < best:
        best = va["total"]
        save(f"{tc['checkpoint_dir']}/{stem}_best.pt", ep)
        print(f"  ** new best (val total={best:.3f}) **")

# -- training curves -----------------------------------------------------------
os.makedirs("outputs", exist_ok=True)
ks = ["policy_nll", "kl", "tawm_state", "tawm_range"]
fig, axes = plt.subplots(1, 5, figsize=(28, 4))
for ax, k in zip(axes, ks):
    ax.plot([h["train"].get(k, float("nan")) for h in hist], label="train")
    ax.plot([h["val"].get(k,   float("nan")) for h in hist], label="val")
    ax.set_title(k); ax.legend()
for k in ["tawm_baseline", "tawm_post", "tawm_prior", "tawm_shuf"]:
    axes[4].plot([h["val"].get(k, float("nan")) for h in hist], label=k)
axes[4].set_title("val TAWM error by z-source"); axes[4].legend()
plt.tight_layout()
out_path = f"outputs/training_curves_{args.trainer}.png"
plt.savefig(out_path, dpi=120)
print(f"saved {out_path}")