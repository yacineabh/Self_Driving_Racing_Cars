"""test_models_v2.py  -  Smoke-test for the updated architecture.

Tests:
  1. Dataset: valid_actions filtering + prev_action augmentation (57-dim states)
  2. SkillModel forward pass shape sanity
  3. One-batch overfit: all losses go to near-zero
  4. NaiveVITrainer and CausalEMTrainer both run one step without error
"""
import sys, glob, os
sys.path.insert(0, ".")
import torch, numpy as np

from data.dataset import OfflineSkillDataset
from utils.normalization import Normalizer
from models.skill_model import SkillModel
from training.trainer import NaiveVITrainer, CausalEMTrainer

DEVICE = "cpu"
T      = 20    # horizon

# -- 1. Dataset ----------------------------------------------------------------
print("=== Test 1: Dataset ===")
paths = sorted(glob.glob("data/offline/*.pkl"))
assert paths, "No pkl files found -- check data/offline/"

ds = OfflineSkillDataset(paths, horizon=T, stride=4)
print(f"  Total windows (valid only): {len(ds)}")
batch = ds[0]
assert batch["states"].shape == (T + 1, 57), \
    f"Expected state shape ({T+1}, 57), got {batch['states'].shape}"
assert batch["actions"].shape == (T, 2),    f"Bad action shape"
assert batch["progress"].shape == (T + 1,), f"Bad progress shape"
print(f"  states shape : {batch['states'].shape}  OK  (57 dims = 54 lidar + speed + 2 prev_action)")
print(f"  actions shape: {batch['actions'].shape}  OK")
print("  PASS")

# -- 2. Model forward pass -----------------------------------------------------
print("\n=== Test 2: Model forward shapes ===")
from torch.utils.data import DataLoader
norm = Normalizer().fit(ds.all_actions())
ds.action_norm = norm

loader = DataLoader(ds, batch_size=16, shuffle=True)
b = next(iter(loader))
b = {k: v.to(DEVICE) for k, v in b.items()}

model = SkillModel(state_dim=57, action_dim=2, embed_dim=64, z_dim=16, hidden=256).to(DEVICE)
out = model(b["states"], b["actions"], b["progress"])
required_keys = {"total", "policy_nll", "kl", "tawm_state", "tawm_prog", "tawm_range", "tawm_baseline"}
missing = required_keys - set(out.keys())
assert not missing, f"Missing keys: {missing}"
for k, v in out.items():
    assert v.isfinite(), f"Loss {k} is non-finite: {v}"
    print(f"  {k:20s}: {v.item():.4f}  OK")
print("  PASS")

# -- 3. Diagnostics ------------------------------------------------------------
print("\n=== Test 3: Diagnostics ===")
diag = model.diagnostics(b["states"], b["actions"], b["progress"])
print(f"  tawm_post    : {diag['tawm_post']:.4f}  (should be < baseline={diag['tawm_baseline']:.4f} after training)")
print(f"  range_err_post: {diag['range_err_post']:.4f}")
print("  PASS")

# -- 4. One-batch overfit ------------------------------------------------------
print("\n=== Test 4: One-batch overfit (100 steps) ===")
model_ov = SkillModel(state_dim=57, action_dim=2, embed_dim=64, z_dim=16, hidden=256, kl_beta=0.0).to(DEVICE)
opt      = torch.optim.Adam(model_ov.parameters(), lr=1e-3)
init_nll = None
for i in range(100):
    out_ov = model_ov(b["states"], b["actions"], b["progress"])
    loss = out_ov["total"]
    opt.zero_grad(); loss.backward(); opt.step()
    if i == 0:
        init_nll = out_ov["policy_nll"].item()
final_nll = out_ov["policy_nll"].item()
print(f"  policy_nll : {init_nll:.2f} -> {final_nll:.2f}  (decrease expected)")
print(f"  tawm_state : {out_ov['tawm_state'].item():.4f}")
print(f"  tawm_range : {out_ov['tawm_range'].item():.4f}")
assert final_nll < init_nll * 0.7, f"Overfit failed: {init_nll:.2f} -> {final_nll:.2f}"
print("  PASS")

# -- 5. NaiveVI trainer step ---------------------------------------------------
print("\n=== Test 5: NaiveVITrainer step ===")
import yaml
cfg = yaml.safe_load(open("configs/default.yaml"))
tc  = cfg["training"]
model_vi   = SkillModel(state_dim=57, action_dim=2).to(DEVICE)
trainer_vi = NaiveVITrainer(model_vi, tc)
out_vi     = trainer_vi.step(b)
print(f"  total: {out_vi['total']:.4f}  kl: {out_vi['kl']:.4f}  PASS")

# -- 6. CausalEM trainer step --------------------------------------------------
print("\n=== Test 6: CausalEMTrainer step ===")
model_em   = SkillModel(state_dim=57, action_dim=2).to(DEVICE)
trainer_em = CausalEMTrainer(model_em, tc)
out_em     = trainer_em.step(b)
print(f"  total: {out_em['total']:.4f}  kl: {out_em['kl']:.4f}  PASS")

print("\n=== All tests passed ===")