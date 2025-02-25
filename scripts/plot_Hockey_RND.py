from math import e
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import glob

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

env = "hockey"

num_to_mod = {
    "145": "DQN",
    "144": "DQN+RND",
    "164": "DQN + RND: curr. (shooting)",
    "174": "DQN + RND: curr. (defense)",
}

plt.figure(figsize=(5, 4))
for i, val in num_to_algo.items():
    file_pattern = f"../saved/agent_{env}_0_{i}.pk"
    matching_files = glob.glob(file_pattern)

    if not matching_files:
        print(f"No matching file for {file_pattern}")
        continue

    filename = matching_files[-1]

    with open(filename, "rb") as f:
        data = pk.load(f)
        episode_rewards = data["episode_rewards"]
        if i == 145:
            plt.plot(running_mean(episode_rewards[:15000],200), label=num_to_algo[i], linewidth=3)
        else:
            plt.plot(running_mean(episode_rewards[:15000],200), label=num_to_algo[i])

plt.xlabel("Training episodes")
plt.ylabel("Episode reward")
plt.legend()
plt.title(f"{env}: RND experiments")
plt.tight_layout()

plt.savefig(f"../plots/{env}_RND.png")
