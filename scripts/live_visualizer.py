"""live_visualizer.py  -  Real-Time Interactive Autonomous Driving Visualizer.

Run this script to watch the SelfRace agent drive in real-time on any track!

Usage:
  python scripts/live_visualizer.py                  # Interactive track selection menu
  python scripts/live_visualizer.py --map Catalunya   # Launch directly on unseen Catalunya
  python scripts/live_visualizer.py --map Spielberg   # Launch on Spielberg
  python scripts/live_visualizer.py --map Austin      # Launch on Austin
  python scripts/live_visualizer.py --map Silverstone # Launch on Silverstone
"""

import sys, os, glob, argparse, yaml, math, time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button

sys.path.insert(0, ".")

from data.collector import SIM_STEPS_PER_TICK, make_env
from utils.track import Track
from utils.lidar import LidarProcessor
from utils.pure_pursuit import MAX_STEER
from utils.normalization import Normalizer
from models.skill_model import SkillModel
from scripts.test_cem_planner import (
    reactive_shield_action,
    make_full_state,
    choose_skill
)


def get_available_maps():
    """Find all track directories in maps/ containing the required map files."""
    maps = []
    if not os.path.exists("maps"):
        return maps
    for d in sorted(os.listdir("maps")):
        full_dir = os.path.join("maps", d)
        if os.path.isdir(full_dir):
            if os.path.exists(os.path.join(full_dir, f"{d}_map.yaml")) and \
               os.path.exists(os.path.join(full_dir, f"{d}_centerline.csv")):
                maps.append(d)
    return maps


def select_map_interactive():
    """Prompt the user with an interactive console menu to choose a track."""
    maps = get_available_maps()
    if not maps:
        sys.exit("Error: No maps found in maps/ folder!")

    print("\n" + "=" * 55)
    print("      SelfRace: REAL-TIME TRACK SELECTOR")
    print("=" * 55)
    for i, m in enumerate(maps, 1):
        unseen_tag = " (UNSEEN TRACK)" if m in ["Catalunya", "Silverstone", "Monza"] else ""
        print(f"  [{i}] {m:<15s}{unseen_tag}")
    print("=" * 55)

    while True:
        try:
            choice = input(f"Select a track number (1-{len(maps)}) [Default 1]: ").strip()
            if choice == "":
                return maps[0]
            idx = int(choice)
            if 1 <= idx <= len(maps):
                return maps[idx - 1]
            print(f"Please enter a number between 1 and {len(maps)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
        except (KeyboardInterrupt, EOFError):
            sys.exit("\nExited.")


def main():
    ap = argparse.ArgumentParser(description="Real-Time Autonomous Driving Visualizer")
    ap.add_argument("--map", default=None, help="Map name to run (e.g. Catalunya, Spielberg, Austin)")
    ap.add_argument("--checkpoint", default="checkpoints/oposm_naive_vi_best.pt")
    ap.add_argument("--normalizer", default="checkpoints/action_normalizer.npz")
    ap.add_argument("--horizon", type=int, default=20, help="Control ticks per skill (20 ticks = 1.0s)")
    ap.add_argument("--fps", type=float, default=20.0, help="Target visual playback FPS")
    args = ap.parse_args()

    # 1. Select Map
    selected_map = args.map or select_map_interactive()
    print(f"\nInitializing SelfRace on: {selected_map}...")

    # 2. Check Device & Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}. Please train the model first!")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["cfg"]
    model = SkillModel(**cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    norm = Normalizer.load(args.normalizer)
    action_mean, action_std = norm.mean, norm.std
    lidar = LidarProcessor()

    # 3. Load Map & Track Geometry
    track = Track("maps", selected_map)
    env = make_env(selected_map)

    map_cfg = yaml.safe_load(open(f"maps/{selected_map}/{selected_map}_map.yaml"))
    map_img = plt.imread(f"maps/{selected_map}/{selected_map}_map.png")
    if map_img.ndim == 3:
        map_img = map_img[..., 0]
    ox, oy = map_cfg["origin"][:2]
    res = map_cfg["resolution"]
    h, w = map_img.shape
    extent = [ox, ox + w * res, oy, oy + h * res]

    # 4. Set up Multi-Panel Matplotlib Figure
    plt.ion()
    fig = plt.figure(figsize=(15, 8.5))
    fig.canvas.manager.set_window_title(f"SelfRace Live: {selected_map} Real-Time Autonomy")

    gs = fig.add_gridspec(3, 3, width_ratios=[2.2, 1.0, 1.0], hspace=0.35, wspace=0.3)

    # Panel 1: Main Top-Down Circuit View
    ax_map = fig.add_subplot(gs[:, 0])
    ax_map.imshow(map_img, cmap="gray", extent=extent, origin="upper")
    ax_map.plot(track.xy[:, 0], track.xy[:, 1], "w--", lw=0.8, alpha=0.4, label="Centerline")
    traj_line, = ax_map.plot([], [], "r-", lw=2.0, label="Car Path")
    car_marker, = ax_map.plot([], [], "go", ms=7, zorder=5)
    car_heading_arrow = None
    lidar_lines = [ax_map.plot([], [], "cyan", lw=0.4, alpha=0.6)[0] for _ in range(lidar.n_beams)]
    ax_map.set_title(f"Circuit: {selected_map} (Zero-Shot Autonomous Execution)", fontsize=12, fontweight="bold")
    ax_map.axis("off")

    # Panel 2: Live LiDAR Polar Radar Scope
    ax_lidar = fig.add_subplot(gs[0, 1:], polar=True)
    ax_lidar.set_theta_zero_location("N")
    ax_lidar.set_theta_direction(-1)
    ax_lidar.set_thetamin(-90)
    ax_lidar.set_thetamax(90)
    lidar_plot, = ax_lidar.plot([], [], "c-", lw=1.5)
    lidar_fill = None
    ax_lidar.set_ylim(0, 10)
    ax_lidar.set_title("Forward LiDAR Radar (180° Field-of-View)", fontsize=10)

    # Panel 3: Speedometer & Steering Angle Display
    ax_controls = fig.add_subplot(gs[1, 1:])
    ax_controls.axis("off")
    bar_speed = ax_controls.barh([1], [0], color="limegreen", height=0.5)
    bar_steer = ax_controls.barh([0], [0], color="dodgerblue", height=0.5)
    ax_controls.set_xlim(-1.2, 6.5)
    ax_controls.set_ylim(-0.5, 1.8)
    text_speed = ax_controls.text(0.1, 1.0, "Speed: 0.0 m/s", va="center", fontsize=11, fontweight="bold")
    text_steer = ax_controls.text(0.1, 0.0, "Steer: 0.00 rad (0.0°)", va="center", fontsize=11, fontweight="bold")
    ax_controls.set_title("Real-Time Control Outputs", fontsize=10)

    # Panel 4: Telemetry & Status HUD
    ax_hud = fig.add_subplot(gs[2, 1:])
    ax_hud.axis("off")
    hud_bg = patches.FancyBboxPatch((0.02, 0.05), 0.96, 0.90, boxstyle="round,pad=0.05",
                                    facecolor="#1e1e24", edgecolor="#3a3a42", transform=ax_hud.transAxes)
    ax_hud.add_patch(hud_bg)
    text_hud_title = ax_hud.text(0.08, 0.80, "TELEMETRY DASHBOARD", color="white", fontsize=11, fontweight="bold")
    text_hud_progress = ax_hud.text(0.08, 0.60, "Distance: 0.0 / 0.0 m (0.0%)", color="#d1d5db", fontsize=10)
    text_hud_mode = ax_hud.text(0.08, 0.40, "Mode: CEM World Model", color="#4ade80", fontsize=10, fontweight="bold")
    text_hud_status = ax_hud.text(0.08, 0.20, "Status: Driving...", color="#93c5fd", fontsize=10)

    # 5. Simulation State
    sim_state = {
        "paused": False,
        "reset_requested": False,
        "quit": False
    }

    def on_key(event):
        if event.key == " ":
            sim_state["paused"] = not sim_state["paused"]
            state_str = "PAUSED" if sim_state["paused"] else "RESUMED"
            print(f"[{state_str}]")
        elif event.key in ["r", "R"]:
            sim_state["reset_requested"] = True
            print("[RESETTING LAP]")
        elif event.key in ["q", "Q", "escape"]:
            sim_state["quit"] = True
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    print("\n" + "-" * 55)
    print(" Controls:")
    print("   [SPACE] : Pause / Resume")
    print("   [R]     : Restart Lap from Start Line")
    print("   [Q/ESC] : Quit Visualizer")
    print("-" * 55 + "\n")

    # 6. Main Live Driving Loop
    while not sim_state["quit"]:
        # Reset car at starting line
        pose = track.start_pose()
        obs, _, _, _ = env.reset(pose)
        init_speed = 3.0
        env.unwrapped.sim.agents[0].state[3] = init_speed
        obs["linear_vels_x"][0] = init_speed

        last_pos_s = track.progress(obs["poses_x"][0], obs["poses_y"][0])
        prev_a_norm = np.zeros(2, dtype=np.float32)

        cumulative_progress = 0.0
        crashed = False
        traj_x = [obs["poses_x"][0]]
        traj_y = [obs["poses_y"][0]]
        total_ticks = 0
        decision_idx = 0

        sim_state["reset_requested"] = False

        while not sim_state["quit"] and not sim_state["reset_requested"]:
            while sim_state["paused"] and not sim_state["quit"] and not sim_state["reset_requested"]:
                plt.pause(0.1)

            if sim_state["quit"] or sim_state["reset_requested"]:
                break

            decision_idx += 1
            state_57 = make_full_state(lidar, obs, prev_a_norm)

            # World Model Planning Decision
            choice, _, _ = choose_skill(
                model=model,
                lidar=lidar,
                state_57=state_57,
                planner="safe_cem",
                plan_horizon=3,
                population=64,
                iterations=5,
                elite_count=8,
                endpoint_threshold_m=0.50,
                risk_floor_m=0.35,
                risk_weight=5.0,
                allow_reactive_fallback=True,
                device=device,
            )

            is_fallback = (choice == "fallback")
            current_mode = "Safety Reflex (Gap)" if is_fallback else "CEM World Model"
            mode_color = "#f59e0b" if is_fallback else "#4ade80"

            # Execute 1-second skill chunk (20 ticks)
            for tick in range(args.horizon):
                if is_fallback:
                    beams = lidar.beams(obs["scans"][0])
                    steer, spd = reactive_shield_action(beams, obs["linear_vels_x"][0])
                    act = np.array([steer, spd])
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

                for _ in range(SIM_STEPS_PER_TICK):
                    obs, _, _, _ = env.step(np.array([[steer, spd]]))
                    if obs["collisions"][0]:
                        crashed = True
                        break

                total_ticks += 1
                car_x = obs["poses_x"][0]
                car_y = obs["poses_y"][0]
                car_th = obs["poses_theta"][0]
                cur_v = obs["linear_vels_x"][0]

                traj_x.append(car_x)
                traj_y.append(car_y)

                # Monotonic centerline progress tracking
                curr_s = track.progress(car_x, car_y)
                ds = curr_s - last_pos_s
                if ds < -track.length / 2: ds += track.length
                elif ds > track.length / 2: ds -= track.length
                cumulative_progress += ds
                last_pos_s = curr_s

                prev_a_norm = norm(np.array([[steer, spd]])).astype(np.float32)[0]

                # Update Visuals Every 2 Ticks for Smooth 20-30 FPS Animation
                if tick % 2 == 0:
                    traj_line.set_data(traj_x, traj_y)
                    car_marker.set_data([car_x], [car_y])

                    # Draw Heading Arrow
                    if car_heading_arrow is not None:
                        car_heading_arrow.remove()
                    car_heading_arrow = ax_map.annotate(
                        "", xy=(car_x + 2.5 * math.cos(car_th), car_y + 2.5 * math.sin(car_th)),
                        xytext=(car_x, car_y),
                        arrowprops=dict(arrowstyle="->", color="yellow", lw=2.5)
                    )

                    # Draw Live LiDAR Rays
                    r = lidar.beams(obs["scans"][0]) * lidar.max_range
                    angles_world = car_th + lidar.angles
                    for b_i in range(lidar.n_beams):
                        end_x = car_x + r[b_i] * math.cos(angles_world[b_i])
                        end_y = car_y + r[b_i] * math.sin(angles_world[b_i])
                        lidar_lines[b_i].set_data([car_x, end_x], [car_y, end_y])

                    # Update Radar Scope
                    lidar_plot.set_data(lidar.angles, r)

                    # Update Gauges & Text
                    text_speed.set_text(f"Speed: {cur_v:.2f} m/s")
                    text_steer.set_text(f"Steer: {steer:+.2f} rad ({math.degrees(steer):+.1f}°)")

                    lap_pct = min(100.0, (cumulative_progress / track.length) * 100.0)
                    text_hud_progress.set_text(
                        f"Distance: {cumulative_progress:.1f} / {track.length:.1f} m ({lap_pct:.1f}%)"
                    )
                    text_hud_mode.set_text(f"Mode: {current_mode}")
                    text_hud_mode.set_color(mode_color)
                    text_hud_status.set_text(
                        f"Time: {total_ticks * 0.05:.1f}s | Decision: #{decision_idx}"
                    )

                    plt.pause(0.001)

                if crashed or cumulative_progress >= track.length:
                    break

            if crashed or cumulative_progress >= track.length:
                break

        # Lap Finished or Crashed
        if cumulative_progress >= track.length:
            text_hud_status.set_text("Status: ** LAP COMPLETED! **")
            text_hud_status.set_color("#4ade80")
            print(f"\n[LAP COMPLETED] Distance: {cumulative_progress:.1f}m in {total_ticks * 0.05:.1f}s!")
        elif crashed:
            text_hud_status.set_text("Status: COLLISION OCCURRED")
            text_hud_status.set_color("#ef4444")
            print("\n[COLLISION] Car hit boundary.")

        plt.pause(2.0)


if __name__ == "__main__":
    main()
