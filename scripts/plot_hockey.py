<<<<<<< HEAD
import numpy as np
import glob
import torch
import matplotlib.pyplot as plt
import pickle as pk
=======
from math import e
import matplotlib.pyplot as plt
import numpy as np
import glob
import torch
>>>>>>> 9a9494bb7e7eb824e142a6f3f5e5f31cae3624c8

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

env = "hockey"

# ALGORITHMS COMPARISON
num_to_algo = {
    1: "DQN",
    3: "DDQN",
    2: "DQN + PER",
    4: "Dueling DQN",
    7: "Dueling DDQN + PER"
}

plt.figure(figsize=(5, 4))
for i, val in num_to_algo.items():
    file_pattern = f"../saved/agent_{env}_*_{i}.pk"
    matching_files = glob.glob(file_pattern)

    if not matching_files:
        print(f"No matching file for {file_pattern}")
        continue

    filename = matching_files[-1]

    try:
        # ✅ Strongest possible fix: Ensure all tensors move to CPU
<<<<<<< HEAD
        data = pk.load(open(filename, 'rb'))
=======
        data = torch.load(filename, map_location=torch.device('cpu'))
>>>>>>> 9a9494bb7e7eb824e142a6f3f5e5f31cae3624c8
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.cpu()  # Ensure every tensor is explicitly on CPU

    except Exception as e:
        print(f"Error loading {filename}: {e}")
        continue

    episode_rewards = data["cum_mean_episode_rewards"]

    if i == 1:
        plt.plot(running_mean(episode_rewards, 200), label=num_to_algo[i], linewidth=3)
    else:
        plt.plot(running_mean(episode_rewards, 200), label=num_to_algo[i])

plt.xlabel("Training episodes")
plt.ylabel("Episode reward")
plt.legend()
plt.title(f"{env}")
plt.tight_layout()
<<<<<<< HEAD
plt.savefig(f"../plots/{env}.png")
=======
plt.savefig(f"../plots/{env}.png")
>>>>>>> 9a9494bb7e7eb824e142a6f3f5e5f31cae3624c8
