"""test_cem_planner.py  -  Closed-Loop Receding-Horizon Skill Planning & Evaluation.

Features:
  - 57-dim temporal state augmentation (55 sensor + 2 normalized prev_action)
  - Integrated TAWM 3-output prediction (delta_state, progress, min_range)
  - Prior-relative CEM sequence optimization (multi-step lookahead)
  - Safety Shield / Reactive Fallback: prevents abstention stalls on tight tracks
    like Spielberg by steering toward the widest LiDAR clearance corridor.
  - Cumulative progress tracking & Full Lap completion detection
  - Trajectory plotting on track map
"""

import sys, os, glob, argparse, yaml, pickle, math
from collections import defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from data.collector import SIM_STEPS_PER_TICK, make_env
from utils.track import Track
from utils.lidar import LidarProcessor
from utils.pure_pursuit import MAX_STEER
from utils.normalization import Normalizer
from models.skill_model import SkillModel, PROGRESS_SCALE, RAW_STATE_DIM, N_LIDAR_BEAMS


# ── Reactive Safety Shield (Follow-The-Gap reflex) ───────────────────────────

def reactive_shield_action(beams_normalized, current_speed, max_speed=2.5):
    """Reflexive steering toward the widest obstacle-free LiDAR corridor.
    Used as safety fallback when all prior skill samples are rejected by the risk floor.
    """
    n_beams = len(beams_normalized)
    angles = np.linspace(-math.pi / 2, math.pi / 2, n_beams)

    # Free threshold: beam must be >= 0.15 normalized (1.5m clearance)
    free = beams_normalized >= 0.15
    if not np.any(free):
        best_idx = np.argmax(beams_normalized)
        return float(np.clip(angles[best_idx], -MAX_STEER, MAX_STEER)), 1.0

    # Find longest contiguous sequence of True (widest gap)
    best_start, best_len = 0, 0
    cur_start, cur_len = 0, 0
    for i, f in enumerate(free):
        if f:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0

    gap_mid = best_start + best_len // 2
    steer = float(np.clip(angles[gap_mid], -MAX_STEER, MAX_STEER))
    clearance = float(beams_normalized[gap_mid])
    target_speed = float(np.clip(clearance * 4.0, 1.2, max_speed))
    return steer, target_speed


# ── Simulator State Helper ───────────────────────────────────────────────────

def make_full_state(lidar, obs, prev_action_norm):
    """Build 57-dim state: 54 LiDAR beams + 1 speed + 2 normalized prev_actions."""
    raw_55 = lidar.state(obs["scans"][0], obs["linear_vels_x"][0])
    return np.concatenate([raw_55, prev_action_norm]).astype(np.float32)


# ── CEM Optimization over Latent Sequences ───────────────────────────────────

def evaluate_latent_population(model, lidar, state_tensor, epsilon,
                               risk_floor_m, risk_weight, device):
    """Roll out population of (candidate_count, horizon, z_dim) through TAWM.
    Returns scores combining predicted progress and safety penalties.
    """
    candidate_count, horizon, z_dim = epsilon.shape
    curr_states = state_tensor.expand(candidate_count, -1).clone()

    seq_z = []
    seq_progress = []
    seq_endpoint_range = []
    seq_min_path_range = []

    with torch.no_grad():
        for step in range(horizon):
            e = model.encoder(curr_states)
            mu_p, ls_p = model.prior(e)
            z = mu_p + ls_p.exp() * epsilon[:, step, :]

            # Integrated TAWM outputs: (delta_raw, progress, min_range)
            delta_raw, prog, min_range = model.tawm(e, z)

            # Update raw state (clamped to physical bounds)
            next_raw = (curr_states[:, :RAW_STATE_DIM] + delta_raw).clamp(0.0, 1.0)

            # End of step min LiDAR range (endpoint safety)
            endpoint_range_m = next_raw[:, :lidar.n_beams].min(dim=1).values * lidar.max_range
            # Min range along the entire skill trajectory
            path_range_m = min_range * lidar.max_range

            # For next step, predict policy's action at endpoint
            pred_a_norm, _ = model.policy(e, z)
            curr_states = torch.cat([next_raw, pred_a_norm], dim=-1)

            seq_z.append(z)
            seq_progress.append(prog * PROGRESS_SCALE)
            seq_endpoint_range.append(endpoint_range_m)
            seq_min_path_range.append(path_range_m)

    progress_all = torch.stack(seq_progress, dim=1)           # (N, horizon)
    endpoint_all = torch.stack(seq_endpoint_range, dim=1)      # (N, horizon)
    path_min_all = torch.stack(seq_min_path_range, dim=1)      # (N, horizon)

    safety_penalty = risk_weight * torch.relu(risk_floor_m - path_min_all).sum(dim=1)
    scores = progress_all.sum(dim=1) - safety_penalty

    return {
        "z": torch.stack(seq_z, dim=1),
        "scores": scores,
        "progress": progress_all,
        "endpoint_range": endpoint_all,
        "path_range": path_min_all,
    }


def choose_skill(model, lidar, state_57, planner, plan_horizon,
                 population, iterations, elite_count,
                 endpoint_threshold_m, risk_floor_m, risk_weight,
                 allow_reactive_fallback, device):
    """Run CEM or random shooting to select the optimal latent skill z."""
    z_dim = model.prior.net[-1].out_features // 2
    state_t = torch.as_tensor(state_57, dtype=torch.float32, device=device).unsqueeze(0)

    if planner == "endpoint_filtered":
        draw_count = population * iterations * plan_horizon
        with torch.no_grad():
            e = model.encoder(state_t)
            mu_p, ls_p = model.prior(e)
            noise = torch.randn((draw_count, z_dim), device=device)
            z_cands = mu_p + ls_p.exp() * noise
            e_rep = e.expand(draw_count, -1)
            delta_raw, prog, min_range = model.tawm(e_rep, z_cands)
            next_raw = (state_t[:, :RAW_STATE_DIM] + delta_raw).clamp(0.0, 1.0)
            endpoint = next_raw[:, :lidar.n_beams].min(dim=1).values * lidar.max_range
            path_min = min_range * lidar.max_range

        eligible = torch.where((endpoint >= endpoint_threshold_m) & (path_min >= risk_floor_m))[0]
        if eligible.numel() == 0:
            eligible = torch.where(endpoint >= endpoint_threshold_m * 0.6)[0]
        if eligible.numel() == 0:
            if allow_reactive_fallback:
                return "fallback", 0, draw_count
            return None, 0, draw_count

        best_i = eligible[torch.argmax(prog[eligible])]
        return {"z": z_cands[best_i:best_i + 1]}, eligible.numel(), draw_count

    # CEM / Safe CEM Sequence Planner
    cem_mean = torch.zeros((plan_horizon, z_dim), device=device)
    cem_std = torch.ones_like(cem_mean)
    best_candidate = None
    total_candidates = population * iterations
    eligible_count = 0

    for it in range(iterations):
        eps = cem_mean.unsqueeze(0) + cem_std.unsqueeze(0) * torch.randn(
            (population, plan_horizon, z_dim), device=device)
        rollout = evaluate_latent_population(
            model, lidar, state_t, eps, risk_floor_m, risk_weight, device)

        valid = (rollout["endpoint_range"][:, 0] >= endpoint_threshold_m) & \
                (rollout["path_range"][:, 0] >= risk_floor_m)
        valid_idx = torch.where(valid)[0]
        eligible_count += valid_idx.numel()

        if valid_idx.numel() > 0:
            valid_scores = rollout["scores"][valid_idx]
            order = torch.argsort(valid_scores, descending=True)
            top_idx = int(valid_idx[order[0]].item())
            if best_candidate is None or rollout["scores"][top_idx] > best_candidate["score"]:
                best_candidate = {
                    "z": rollout["z"][top_idx, 0:1],
                    "score": float(rollout["scores"][top_idx].cpu()),
                    "first_progress": float(rollout["progress"][top_idx, 0].cpu()),
                }

            elite_idx = valid_idx[order[:min(elite_count, len(order))]]
            elites = eps[elite_idx]
            cem_mean = elites.mean(dim=0)
            cem_std = elites.std(dim=0, unbiased=False).clamp(0.15, 1.5)

    if best_candidate is None:
        if allow_reactive_fallback:
            return "fallback", eligible_count, total_candidates
        return None, eligible_count, total_candidates

    return best_candidate, eligible_count, total_candidates


# ── Plotting Helper ──────────────────────────────────────────────────────────

def plot_lap(map_name, track, traj_x, traj_y, save_path):
    """Plot car trajectory on the racetrack map."""
    cfg = yaml.safe_load(open(f"maps/{map_name}/{map_name}_map.yaml"))
    img = plt.imread(f"maps/{map_name}/{map_name}_map.png")
    if img.ndim == 3:
        img = img[..., 0]
    ox, oy = cfg["origin"][:2]
    res = cfg["resolution"]
    h, w = img.shape
    extent = [ox, ox + w * res, oy, oy + h * res]

    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap="gray", extent=extent, origin="upper")
    plt.plot(track.xy[:, 0], track.xy[:, 1], "w--", lw=1, alpha=0.5, label="Centerline")
    plt.plot(traj_x, traj_y, "r-", lw=2, label="SelfRace Agent")
    plt.plot(traj_x[0], traj_y[0], "go", ms=8, label="Start")
    plt.plot(traj_x[-1], traj_y[-1], "bo", ms=8, label="End")
    plt.title(f"Zero-Shot Lap Trajectory - {map_name}", fontsize=14)
    plt.legend(loc="upper right")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved trajectory plot to {save_path}")


# ── Closed-Loop Evaluation Loop ──────────────────────────────────────────────

def evaluate_map(map_name, model, normalizer, lidar, track, env,
                 n_episodes=1, max_decisions=100, horizon=20,
                 full_lap_mode=False, planner="safe_cem", plan_horizon=3,
                 population=64, iterations=5, elite_count=8,
                 endpoint_threshold_m=0.50, risk_floor_m=0.35, risk_weight=5.0,
                 allow_reactive_fallback=True, save_plot=False, device="cpu", seed=42):
    rng = np.random.default_rng(seed)
    results = []

    action_mean = normalizer.mean
    action_std  = normalizer.std

    for ep_i in range(n_episodes):
        if full_lap_mode:
            # Start exactly at track start line
            pose = track.start_pose()
            obs, _, _, _ = env.reset(pose)
            init_speed = 3.0
        else:
            # Random start along centerline
            idx = rng.integers(track.n)
            p, q = track.xy[idx], track.xy[(idx + 1) % track.n]
            heading = np.arctan2(q[1] - p[1], q[0] - p[0])
            norm = np.array([-np.sin(heading), np.cos(heading)])
            half_w = min(track.w_right[idx], track.w_left[idx])
            pos = p + rng.uniform(-0.3, 0.3) * half_w * norm
            yaw = heading + rng.uniform(-0.1, 0.1)

            obs, _, _, _ = env.reset(np.array([[pos[0], pos[1], yaw]]))
            init_speed = float(rng.uniform(2.5, 3.5))

        env.unwrapped.sim.agents[0].state[3] = init_speed
        obs["linear_vels_x"][0] = init_speed

        last_pos_s = track.progress(obs["poses_x"][0], obs["poses_y"][0])
        prev_a_norm = np.zeros(2, dtype=np.float32)

        cumulative_progress = 0.0
        crashed = False
        abstained = False
        fallbacks = 0
        decisions_executed = 0
        total_ticks = 0

        traj_x = [obs["poses_x"][0]]
        traj_y = [obs["poses_y"][0]]

        for d in range(max_decisions):
            state_57 = make_full_state(lidar, obs, prev_a_norm)

            choice, n_elig, n_total = choose_skill(
                model, lidar, state_57, planner, plan_horizon,
                population, iterations, elite_count,
                endpoint_threshold_m, risk_floor_m, risk_weight,
                allow_reactive_fallback, device)

            if choice is None:
                abstained = True
                break

            if choice == "fallback":
                fallbacks += 1
                for _ in range(horizon):
                    beams = lidar.beams(obs["scans"][0])
                    steer, spd = reactive_shield_action(beams, obs["linear_vels_x"][0])
                    for _ in range(SIM_STEPS_PER_TICK):
                        obs, _, _, _ = env.step(np.array([[steer, spd]]))
                        if obs["collisions"][0]:
                            crashed = True
                            break
                    total_ticks += 1
                    traj_x.append(obs["poses_x"][0])
                    traj_y.append(obs["poses_y"][0])

                    # Step progress accumulation
                    curr_s = track.progress(obs["poses_x"][0], obs["poses_y"][0])
                    ds = curr_s - last_pos_s
                    if ds < -track.length / 2: ds += track.length
                    elif ds > track.length / 2: ds -= track.length
                    cumulative_progress += ds
                    last_pos_s = curr_s

                    if crashed:
                        break
                    prev_a_norm = normalizer(np.array([[steer, spd]])).astype(np.float32)[0]
                decisions_executed += 1
                if crashed or (full_lap_mode and cumulative_progress >= track.length):
                    break
                continue

            # Execute world model policy with chosen latent z
            z = choice["z"]
            for _ in range(horizon):
                st_57 = make_full_state(lidar, obs, prev_a_norm)
                st_t = torch.as_tensor(st_57, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    e = model.encoder(st_t)
                    a_norm_mu, _ = model.policy(e, z)
                a_norm = a_norm_mu[0].cpu().numpy()
                act = a_norm * action_std + action_mean
                steer = float(np.clip(act[0], -MAX_STEER, MAX_STEER))
                spd = float(np.clip(act[1], 0.5, lidar.v_max))

                for _ in range(SIM_STEPS_PER_TICK):
                    obs, _, _, _ = env.step(np.array([[steer, spd]]))
                    if obs["collisions"][0]:
                        crashed = True
                        break
                total_ticks += 1
                traj_x.append(obs["poses_x"][0])
                traj_y.append(obs["poses_y"][0])

                curr_s = track.progress(obs["poses_x"][0], obs["poses_y"][0])
                ds = curr_s - last_pos_s
                if ds < -track.length / 2: ds += track.length
                elif ds > track.length / 2: ds -= track.length
                cumulative_progress += ds
                last_pos_s = curr_s

                if crashed:
                    break
                prev_a_norm = a_norm

            decisions_executed += 1
            if crashed or (full_lap_mode and cumulative_progress >= track.length):
                break

        lap_completed = cumulative_progress >= track.length
        time_s = total_ticks * 0.05
        avg_speed = cumulative_progress / max(0.1, time_s)

        results.append({
            "episode": ep_i,
            "progress_m": cumulative_progress,
            "crashed": crashed,
            "abstained": abstained,
            "fallbacks": fallbacks,
            "decisions": decisions_executed,
            "lap_completed": lap_completed,
            "time_s": time_s,
            "avg_speed": avg_speed,
        })

        if save_plot and ep_i == 0:
            os.makedirs("outputs", exist_ok=True)
            plot_path = f"outputs/trajectory_{map_name}.png"
            plot_lap(map_name, track, traj_x, traj_y, plot_path)

        if full_lap_mode:
            lap_pct = min(100.0, cumulative_progress / track.length * 100)
            status_str = "COMPLETED" if lap_completed else ("CRASHED" if crashed else "STOPPED")
            print(f"[{map_name}] Lap {ep_i+1}: {status_str} | Dist: {cumulative_progress:.1f}/{track.length:.1f}m ({lap_pct:.1f}%) | Time: {time_s:.1f}s | Speed: {avg_speed:.2f}m/s | Fallbacks: {fallbacks}")

    prog_list = [r["progress_m"] for r in results]
    crash_rate = np.mean([r["crashed"] for r in results])
    abstain_rate = np.mean([r["abstained"] for r in results])
    fallback_rate = np.mean([r["fallbacks"] > 0 for r in results])
    mean_prog = np.mean(prog_list)
    max_prog = np.max(prog_list)
    laps_completed = sum(r["lap_completed"] for r in results)

    print(f"\n[{map_name}] Summary ({n_episodes} episodes, planner={planner}):")
    print(f"  Mean Progress:   {mean_prog:.2f} m  (max {max_prog:.2f} m)")
    print(f"  Collision Rate:  {crash_rate:.1%}")
    print(f"  Abstain Rate:    {abstain_rate:.1%}")
    print(f"  Shield Used:     {fallback_rate:.1%}")
    if full_lap_mode:
        print(f"  Laps Completed:  {laps_completed}/{n_episodes} ({laps_completed/n_episodes:.1%})")

    return {
        "map": map_name,
        "mean_progress": mean_prog,
        "max_progress": max_prog,
        "collision_rate": crash_rate,
        "abstain_rate": abstain_rate,
        "fallback_rate": fallback_rate,
        "laps_completed": laps_completed,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/oposm_naive_vi_best.pt")
    ap.add_argument("--normalizer", default="checkpoints/action_normalizer.npz")
    ap.add_argument("--maps", nargs="*", default=["Spielberg", "Austin", "MoscowRaceway"])
    ap.add_argument("--planner", choices=["safe_cem", "cem", "endpoint_filtered"], default="safe_cem")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--decisions", type=int, default=15)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--full-lap", action="store_true", help="Attempt full lap from start line")
    ap.add_argument("--save-plot", action="store_true", help="Save trajectory plot image to outputs/")
    ap.add_argument("--no-shield", action="store_true", help="Disable reactive safety shield")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint {args.checkpoint} on {device}...")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["cfg"]
    model = SkillModel(**cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    norm = Normalizer.load(args.normalizer)
    lidar = LidarProcessor()

    # For full lap mode, ensure enough decisions (e.g. 150 = 150s = ~600m)
    decisions = max(args.decisions, 150) if args.full_lap else args.decisions

    summaries = []
    for map_name in args.maps:
        print(f"\n--- Testing on {map_name} ---")
        track = Track("maps", map_name)
        env = make_env(map_name)
        res = evaluate_map(
            map_name=map_name,
            model=model,
            normalizer=norm,
            lidar=lidar,
            track=track,
            env=env,
            n_episodes=args.episodes,
            max_decisions=decisions,
            horizon=args.horizon,
            full_lap_mode=args.full_lap,
            planner=args.planner,
            allow_reactive_fallback=not args.no_shield,
            save_plot=args.save_plot,
            device=device,
        )
        summaries.append(res)

    print("\n================ FINAL BENCHMARK SUMMARY ================")
    for s in summaries:
        lap_info = f" | Full Laps: {s.get('laps_completed', 0)}" if args.full_lap else ""
        print(f"Map: {s['map']:15s} | Prog: {s['mean_progress']:6.2f}m | "
              f"Col: {s['collision_rate']:5.1%} | Abstain: {s['abstain_rate']:5.1%} | "
              f"Shield: {s['fallback_rate']:5.1%}{lap_info}")


if __name__ == "__main__":
    main()
