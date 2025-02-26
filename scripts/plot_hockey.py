from math import e
import matplotlib.pyplot as plt
import numpy as np
import glob
import torch
import pickle as pk

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

env = "hockey"

# ALGORITHMS COMPARISON
num_to_algo = {
    "1_20000": "DQN",
    "3_20000": "DDQN",
    "2_20000": "DQN + PER",
    "4_20000": "Dueling DQN",
    "6": "3-step DQN",
    "8": "DQN + RND",
    "10": "Dueling DQN + PER",
    "7_20000": "Dueling DDQN + PER"
}

plt.figure(figsize=(5, 4))
for i, val in num_to_algo.items():
    file_pattern = f"../saved/agent_{env}_*_{i}.pk"
    matching_files = glob.glob(file_pattern)

    if not matching_files:
        print(f"No matching file for {file_pattern}")
        continue

    filename = matching_files[0]

    try:
        data = pk.load(open(filename, "rb"))

    except Exception as e:
        print(f"Error loading {filename}: {e}")
        continue

    episode_rewards = data["cum_mean_episode_rewards"]
    episode_wins = data["episode_wins"]
    # if i == "10":
    #     indices = np.linspace(0, len(episode_rewards)-1, 15000).astype(int)
    #     episode_rewards = np.array(episode_rewards)[indices]
    #     episode_wins = np.array(episode_wins)[indices]

    if i == "1_20000":
        plt.plot(running_mean(episode_wins, 1000), label=num_to_algo[i], linewidth=3)
    else:
        plt.plot(running_mean(episode_wins, 1000), label=num_to_algo[i], linewidth=1)

plt.xlabel("Training episodes")
plt.ylabel("Fraction wins")
# plt.ylabel("Episode reward")
plt.legend(prop={'size': 8}, borderpad=0.3)
plt.tight_layout()
plt.savefig(f"../plots/{env}2.png")