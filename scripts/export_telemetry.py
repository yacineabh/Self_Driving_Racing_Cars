"""export_telemetry.py  -  Export full lap telemetry & LiDAR data to JSON for interactive web visualizer."""

import sys, os, glob, yaml, json, math
import numpy as np
import torch

sys.path.insert(0, ".")

from data.collector import SIM_STEPS_PER_TICK, make_env
from utils.track import Track
from utils.lidar import LidarProcessor
from utils.pure_pursuit import MAX_STEER
from utils.normalization import Normalizer
from models.skill_model import SkillModel
from scripts.test_cem_planner import reactive_shield_action, make_full_state, choose_skill


def export_track_data(map_name, model, norm, lidar, device, max_decisions=150):
    print(f"Exporting {map_name}...")
    track = Track("maps", map_name)
    env = make_env(map_name)

    map_cfg = yaml.safe_load(open(f"maps/{map_name}/{map_name}_map.yaml"))
    ox, oy = map_cfg["origin"][:2]
    res = map_cfg["resolution"]

    action_mean, action_std = norm.mean, norm.std

    pose = track.start_pose()
    obs, _, _, _ = env.reset(pose)
    init_speed = 3.0
    env.unwrapped.sim.agents[0].state[3] = init_speed
    obs["linear_vels_x"][0] = init_speed

    last_pos_s = track.progress(obs["poses_x"][0], obs["poses_y"][0])
    prev_a_norm = np.zeros(2, dtype=np.float32)

    cumulative_progress = 0.0
    frames = []

    # Centerline subsampled for lightweight web rendering
    centerline = track.xy[::2].tolist()

    for d in range(max_decisions):
        state_57 = make_full_state(lidar, obs, prev_a_norm)
        choice, _, _ = choose_skill(
            model=model, lidar=lidar, state_57=state_57,
            planner="safe_cem", plan_horizon=3, population=64,
            iterations=5, elite_count=8, endpoint_threshold_m=0.50,
            risk_floor_m=0.35, risk_weight=5.0, allow_reactive_fallback=True,
            device=device
        )

        is_fallback = (choice == "fallback")
        mode = "Safety Reflex" if is_fallback else "World Model"

        for tick in range(20):
            if is_fallback:
                beams = lidar.beams(obs["scans"][0])
                steer, spd = reactive_shield_action(beams, obs["linear_vels_x"][0])
            else:
                z = choice["z"]
                st_57 = make_full_state(lidar, obs, prev_a_norm)
                st_t = torch.as_tensor(st_57, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    e = model.encoder(st_t)
                    a_norm_mu, _ = model.policy(e, z)
                a_norm = a_norm_mu[0].cpu().numpy()
                act = a_norm * action_std + action_mean
                steer = float(np.clip(act[0], -MAX_STEER, MAX_STEER))
                spd = float(np.clip(act[1], 0.5, lidar.v_max))

            crashed = False
            for _ in range(SIM_STEPS_PER_TICK):
                obs, _, _, _ = env.step(np.array([[steer, spd]]))
                if obs["collisions"][0]:
                    crashed = True
                    break

            car_x = float(obs["poses_x"][0])
            car_y = float(obs["poses_y"][0])
            car_th = float(obs["poses_theta"][0])
            cur_v = float(obs["linear_vels_x"][0])

            curr_s = track.progress(car_x, car_y)
            ds = curr_s - last_pos_s
            if ds < -track.length / 2: ds += track.length
            elif ds > track.length / 2: ds -= track.length
            cumulative_progress += ds
            last_pos_s = curr_s

            prev_a_norm = norm(np.array([[steer, spd]])).astype(np.float32)[0]

            # Save 1 frame every 2 ticks (10 Hz = smooth 0.1s steps, light JSON)
            if tick % 2 == 0:
                raw_beams = lidar.beams(obs["scans"][0]) * lidar.max_range
                # Store 18 subsampled beams for light web rendering
                sub_beams = [round(float(b), 2) for b in raw_beams[::3]]
                frames.append({
                    "x": round(car_x, 2),
                    "y": round(car_y, 2),
                    "th": round(car_th, 3),
                    "v": round(cur_v, 2),
                    "steer": round(steer, 3),
                    "prog": round(cumulative_progress, 1),
                    "mode": mode,
                    "beams": sub_beams
                })

            if crashed or cumulative_progress >= track.length:
                break
        if crashed or cumulative_progress >= track.length:
            break

    return {
        "name": map_name,
        "length": round(track.length, 1),
        "origin": [ox, oy],
        "resolution": res,
        "centerline": centerline,
        "frames": frames
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load("checkpoints/oposm_naive_vi_best.pt", map_location=device)
    model = SkillModel(**ckpt["cfg"]["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    norm = Normalizer.load("checkpoints/action_normalizer.npz")
    lidar = LidarProcessor()

    tracks_data = {}
    maps = ["Spielberg", "Austin", "MoscowRaceway", "Catalunya"]
    for m in maps:
        tracks_data[m] = export_track_data(m, model, norm, lidar, device)

    os.makedirs("outputs", exist_ok=True)
    out_file = "outputs/tracks_telemetry.json"
    with open(out_file, "w") as f:
        json.dump(tracks_data, f)
    print(f"Exported all tracks telemetry to {out_file} ({os.path.getsize(out_file) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
