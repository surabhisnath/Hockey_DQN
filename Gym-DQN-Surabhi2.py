import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import torch
from feedforward import DQNAgent
import pylab as plt

# %matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import hockey.hockey_env as h_env


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# env_name = 'Pendulum-v1'
env_name = "CartPole-v0"
env = gym.make(env_name)

# env = h_env.HockeyEnv()

# if isinstance(env.action_space, spaces.Box):
#     print("Discretizing actions")
#     env = DiscreteActionWrapper(env,5)

ac_space = env.action_space
o_space = env.observation_space

q_agent = DQNAgent(
    o_space, ac_space, eps=0.2, update_Qt_after=20, PrioritizedMemory=True
)


episode_rewards = []
cum_mean_episode_rewards = []
losses = []
max_episodes = 8000
max_steps = 500
printevery = 1000
num_stored = 0
for i in range(max_episodes):
    # print(f"Starting episode {i+1}")
    ob, _info = env.reset()
    total_reward = 0

    for t in range(max_steps):
        a = q_agent.act(ob)
        (ob_new, reward, done, trunc, _info) = env.step(a)

        total_reward += reward
        q_agent.store_transition((ob, a, reward, ob_new, done))
        num_stored += 1
        ob = ob_new

        if done:
            break

    # print(f"Episode {i+1} ended after {t+1} steps. Episode reward = {total_reward}")

    episode_rewards.append(total_reward)
    cum_mean_episode_rewards.append(np.mean(episode_rewards[-printevery:]))
    losses.append(np.mean(q_agent.train(1)))

    if (i + 1) % printevery == 0:
        print(
            f"{i+1} episodes completed: Mean cumulative reward: {np.mean(episode_rewards[-printevery:])}"
        )
