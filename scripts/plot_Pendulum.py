from math import e
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import glob

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

env = "Pendulum-v1"

# ALGORITHMS COMPARISON
num_to_algo = {
    "1": "DQN",
    "3": "DDQN",
    "2": "DQN + PER",
    "4": "Dueling DQN",
    "8": "DQN + RND",
    "10": "Dueling DQN + PER",
    "7": "Dueling DDQN + PER"
}

plt.figure(figsize=(5, 4))
for i, val in num_to_algo.items():
    file_pattern = f"../saved/agent_{env}_*_{i}.pk"
    matching_files = glob.glob(file_pattern)

    if not matching_files:
        print(f"No matching file for {file_pattern}")
        continue

    filename = matching_files[0]

    with open(filename, "rb") as f:
        data = pk.load(f)
        if i == "8":
            episode_rewards = data["train_rewards"][:,1]
        else:
            if "8" in num_to_algo:
                episode_rewards = data["episode_rewards"]
            else:
                episode_rewards = data["cum_mean_episode_rewards"]

        if i == "1":
            plt.plot(running_mean(episode_rewards[:1200],200), label=num_to_algo[i], linewidth=3)
        else:
            plt.plot(running_mean(episode_rewards[:1200],200), label=num_to_algo[i])

plt.xlabel("Training episodes")
plt.ylabel("Episode reward")
plt.legend()
plt.title(f"{env}")
plt.tight_layout()
plt.savefig(f"../plots/{env}_all.png")


# MULTISTEP ANALYSIS
num_to_algo = {
    "1": "DQN",
    "63": "3-step DQN",
    "65": "5-step DQN",
    "610": "10-step DQN",
    "6100": "100-step DQN",
    "5": "MC DQN",
}

plt.figure(figsize=(5, 4))
for i, val in num_to_algo.items():
    file_pattern = f"../saved/agent_{env}_*_{i}.pk"
    matching_files = glob.glob(file_pattern)

    if not matching_files:
        print(f"No matching file for {file_pattern}")
        continue

    filename = matching_files[0]

    with open(filename, "rb") as f:
        data = pk.load(f)
        episode_rewards = data["cum_mean_episode_rewards"]
        print(i, episode_rewards[-1])
        if i == "1":
            plt.plot(running_mean(episode_rewards[:1200],200), label=num_to_algo[i], linewidth=3)
        else:
            plt.plot(running_mean(episode_rewards[:1200],200), label=num_to_algo[i])

plt.xlabel("Training episodes")
plt.ylabel("Episode reward")
plt.legend(prop={'size': 8}, borderpad=0.3)
plt.title(f"{env}")
plt.tight_layout()
plt.savefig(f"../plots/{env}_multistep.png")