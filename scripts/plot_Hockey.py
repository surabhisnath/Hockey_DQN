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
    "1": "DQN",
    "3": "DDQN",
    "2": "DQN + PER",
    "4": "Dueling DQN",
    "6": "3-step DQN",
    "8": "DQN + RND",
    "10": "Dueling DQN + PER",
    "7": "Dueling DDQN + PER"
}


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

    if i == "1":
        plt.plot(running_mean(episode_rewards, 1000), label=num_to_algo[i], linewidth=3)
    else:
        plt.plot(running_mean(episode_rewards, 1000), label=num_to_algo[i], linewidth=1)

plt.xlabel("Training episodes")
plt.ylabel("Episode reward")
plt.legend(prop={'size': 8}, borderpad=0.3)
plt.tight_layout()
plt.savefig(f"../plots/H{env[1:]}_episode_rewards.png")


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

    episode_wins = data["episode_wins"]

    if i == "1":
        plt.plot(running_mean(episode_wins, 1000), label=num_to_algo[i], linewidth=3)
    else:
        plt.plot(running_mean(episode_wins, 1000), label=num_to_algo[i], linewidth=1)

plt.xlabel("Training episodes")
plt.ylabel("Fraction wins")
plt.legend(prop={'size': 8}, borderpad=0.3)
plt.tight_layout()
plt.savefig(f"../plots/H{env[1:]}_fraction_wins.png")