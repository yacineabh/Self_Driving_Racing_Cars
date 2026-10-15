import sys, gym, numpy as np
sys.path.insert(0, ".")
from utils.track import Track
from utils.pure_pursuit import PurePursuit

def run(name="Spielberg", speed=3.0, render=True, max_steps=20000):
    track = Track("maps", name)
    env = gym.make("f110_gym:f110-v0", map=f"maps/{name}/{name}_map",
                   map_ext=".png", num_agents=1)
    obs, _, done, _ = env.reset(track.start_pose())
    pp = PurePursuit(track, speed=speed)

    total, last_s = 0.0, track.progress(obs["poses_x"][0], obs["poses_y"][0])
    for t in range(max_steps):
        x, y, th = obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]
        v = obs["linear_vels_x"][0]
        steer, spd = pp.act(x, y, th, v)

        for _ in range(5):                       # 5 sim steps (0.01s) = one 20 Hz control tick
            obs, _, done, _ = env.step(np.array([[steer, spd]]))
        if render: env.render(mode="human")

        s = track.progress(obs["poses_x"][0], obs["poses_y"][0])
        ds = s - last_s
        if ds < -track.length / 2: ds += track.length   # wrapped past the start line
        total += ds; last_s = s

        if obs["collisions"][0]:
            print(f"{name}: COLLISION at {total:.1f}/{track.length:.1f} m, speed={speed}")
            return False
        if total >= track.length:
            print(f"{name}: LAP DONE in {t*0.05:.1f}s, speed={speed}")
            return True
    print(f"{name}: timeout"); return False

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "Spielberg", float(sys.argv[2]) if len(sys.argv) > 2 else 3.0)