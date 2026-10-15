import gym, numpy as np

MAP = "maps/Spielberg/Spielberg_map"   # no extension

env = gym.make("f110_gym:f110-v0", map=MAP, map_ext=".png", num_agents=1)

# Start at the first centerline point, heading along the track
cl = np.loadtxt("maps/Spielberg/Spielberg_centerline.csv", delimiter=",", skiprows=1)
x, y = cl[0, 0], cl[0, 1]
theta = np.arctan2(cl[1, 1] - y, cl[1, 0] - x)

obs, _, done, _ = env.reset(np.array([[x, y, theta]]))
print("obs keys:", obs.keys())
print("scan shape:", np.array(obs["scans"]).shape)

for _ in range(300):
    obs, _, done, _ = env.step(np.array([[0.0, 2.0]]))   # [steer, speed]
    env.render(mode="human")
    if done:
        break