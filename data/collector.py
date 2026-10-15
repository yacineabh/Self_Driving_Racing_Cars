import numpy as np, gym
from utils.track import Track
from utils.pure_pursuit import PurePursuit, MAX_STEER
from utils.lidar import LidarProcessor

SIM_STEPS_PER_TICK = 5   # 5 x 0.01 s = 0.05 s -> 20 Hz control

def make_env(name):
    return gym.make("f110_gym:f110-v0", map=f"maps/{name}/{name}_map",
                    map_ext=".png", num_agents=1)

def random_start(track, rng, lat_frac, yaw_noise):
    i = rng.integers(track.n)
    p, q = track.xy[i], track.xy[(i + 1) % track.n]
    heading = np.arctan2(q[1] - p[1], q[0] - p[0])
    normal = np.array([-np.sin(heading), np.cos(heading)])
    half = min(track.w_right[i], track.w_left[i])
    pos = p + rng.uniform(-lat_frac, lat_frac) * half * normal
    yaw = heading + rng.uniform(-yaw_noise, yaw_noise)
    return np.array([[pos[0], pos[1], yaw]])

def run_episode(env, track, lidar, rng, cfg):
    speed = rng.uniform(*cfg["speed_range"])
    noise_scale = rng.uniform(*cfg.get("noise_scale_range", [1.0, 1.0]))
    pp = PurePursuit(track, speed=speed)
    obs, _, _, _ = env.reset(random_start(track, rng, cfg["start_lateral_frac"],
                                          cfg["start_yaw_noise"]))
    S, A, P = [], [], []
    valid = []          # True = full 50 ms tick completed (safe to use in training)
    noise, crashed = 0.0, False

    def read(o):
        return (o["poses_x"][0], o["poses_y"][0], o["poses_theta"][0],
                o["linear_vels_x"][0])

    for _ in range(cfg["max_steps"]):
        x, y, th, v = read(obs)
        S.append(lidar.state(obs["scans"][0], v))
        P.append([x, y, th])

        steer, spd = pp.act(x, y, th, v)          # clean Pure-Pursuit command
        noise = ((1 - cfg["steer_noise_theta"]) * noise
                 + noise_scale * cfg["steer_noise_std"] * rng.standard_normal())
        steer = float(np.clip(steer + noise, -MAX_STEER, MAX_STEER))
        spd   = float(max(0.5, spd + noise_scale * cfg["speed_noise_std"]
                          * rng.standard_normal()))

        # Advance one control tick; check for collision at every sub-step.
        tick_complete = True
        for _ in range(SIM_STEPS_PER_TICK):
            obs, _, _, _ = env.step(np.array([[steer, spd]]))
            if obs["collisions"][0]:
                crashed = True
                tick_complete = False
                break

        A.append([steer, spd])        # store executed command regardless
        valid.append(tick_complete)   # only True when all 5 sub-steps completed cleanly
        if not tick_complete:
            break                     # end episode at first sub-step collision

    x, y, th, v = read(obs)          # final state -> L+1 states total
    S.append(lidar.state(obs["scans"][0], v))
    P.append([x, y, th])

    # Cumulative centerline progress (handles lap wrap-around)
    s = np.array([track.progress(px, py) for px, py, _ in P])
    ds = np.diff(s)
    ds = np.where(ds < -track.length / 2, ds + track.length, ds)
    ds = np.where(ds >  track.length / 2, ds - track.length, ds)
    progress = np.concatenate([[0.0], np.cumsum(ds)])

    # Build prev_actions: action[t-1] for each tick t (zeros at t=0)
    A_arr = np.array(A, np.float32)
    prev_A = np.zeros_like(A_arr)
    prev_A[1:] = A_arr[:-1]

    return {
        "states":        np.array(S,     np.float32),   # (L+1, 55)
        "actions":       A_arr,                          # (L,   2)
        "prev_actions":  prev_A,                         # (L,   2)  <-- NEW
        "valid_actions": np.array(valid, bool),          # (L,)       <-- NEW
        "poses":         np.array(P,     np.float32),    # (L+1, 3)
        "progress":      progress.astype(np.float32),    # (L+1,)
        "crashed":       crashed,
        "target_speed":  speed,
        "noise_scale":   noise_scale,
    }

def collect_map(name, n_episodes, cfg, seed):
    rng = np.random.default_rng(seed)
    track, lidar, env = Track("maps", name), LidarProcessor(), make_env(name)
    episodes = []
    for k in range(n_episodes):
        ep = run_episode(env, track, lidar, rng, cfg)
        complete = int(ep["valid_actions"].sum())
        episodes.append(ep)
        print(f"[{name}] ep {k+1}/{n_episodes}  "
              f"valid_ticks={complete}/{len(ep['actions'])}"
              f"  speed={ep['target_speed']:.1f}  crashed={ep['crashed']}", flush=True)
    return episodes