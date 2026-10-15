import sys, os, pickle, argparse, yaml
sys.path.insert(0, ".")
from data.collector import collect_map

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--episodes", type=int, default=None, help="override episodes_per_map")
ap.add_argument("--maps", nargs="*", default=None)
ap.add_argument("--tag", default="")
ap.add_argument("--seed_offset", type=int, default=0)
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config))
c = cfg["collection"]
n_ep = args.episodes or c["episodes_per_map"]
os.makedirs(c["out_dir"], exist_ok=True)

for i, name in enumerate(args.maps or c["map_names"]):
    eps = collect_map(name, n_ep, c, seed=cfg["seed"] + i + args.seed_offset)
    suffix = f"_{args.tag}" if args.tag else ""
    path = f"{c['out_dir']}/{name}{suffix}.pkl"
    pickle.dump({"map": name, "episodes": eps}, open(path, "wb"))
    print("saved", path)