from pickletools import optimize
import gymnasium as gym
from gymnasium import spaces
from gymnasium import *
import numpy as np
import time
import torch
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import argparse
from collections import Counter
from agent import DQNAgent
import pickle as pk
from matplotlib import animation
from PIL import Image
import sys
import os
# import imageio.v2 as imageio
sys.path.append(os.path.abspath("../"))
import hockey.hockey_env as h_env

class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, bins=5):
        """A wrapper for converting a 1D continuous actions into discrete ones.
        Args:
            env: The environment to apply the wrapper
            bins: number of discrete actions
        """
        assert isinstance(env.action_space, spaces.Box)
        super().__init__(env)
        self.bins = bins
        self.orig_action_space = env.action_space
        self.action_space = spaces.Discrete(self.bins)

    def action(self, action):
        """discrete actions from low to high in 'bins'
        Args:
            action: The discrete action
        Returns:
            continuous action
        """
        return self.orig_action_space.low + action / (self.bins - 1.0) * (
            self.orig_action_space.high - self.orig_action_space.low
        )

def test_agent(config):
 
    filename = config["filename"]
    render_mode = "human" if config["render"] else None
    save_gif = config["savegif"]
    envname = config["env"]

    if save_gif:
        if render_mode is not None:
            print("overwrite render-mode to image")
        render_mode = "rgb_array"
        
        os.makedirs('../gifs', exist_ok=True)
        print("to get a gif for episode 1 run \"convert -delay 1x30 ../gifs/01_* ep01.gif\"")

    # create environment
    if envname == "hockey":
        env = h_env.HockeyEnv()
        # env.render(mode=render_mode)
        env.discretize_actions(config["numdiscreteactions"])
    else:
        env = gym.make(envname, render_mode = render_mode)
        if isinstance(env.action_space, spaces.Box):
            env = DiscreteActionWrapper(env, config["numdiscreteactions"])

    # create and load agent
    agent = DQNAgent(env.observation_space, env.action_space, config)
    agent.Q.load_state_dict(torch.load("../saved/" + filename))

    print("OPPONENT", config["opponent"])
    # opponent for hockey
    if envname == "hockey" and config["opponent"] == "weak":
        opponent = h_env.BasicOpponent(weak=True)
    if envname == "hockey" and config["opponent"] == "strong":
        opponent = h_env.BasicOpponent(weak=False)
    if envname == "hockey" and config["opponent"] == "self":
        opponent = agent

    # test agent
    test_stats = []
    frames = []
    if envname == "hockey":
        wins = []
    for i in range(config["numtestepisodes"]):
        ob, _ = env.reset()
        if envname == "hockey":
            ob2 = env.obs_agent_two()
        total_reward = 0
        for t in range(config["numsteps"]):
            done = False
            a = agent.act(ob, 0)
            if envname == "hockey":
                a1 = env.action(a)
                a2 = opponent.act(ob2)
                (ob_new, reward, done, _, info) = env.step(np.hstack([a1,a2]))
            else:
                (ob_new, reward, done, _, _) = env.step(a)
            ob=ob_new
            total_reward+= reward
            if save_gif and i < 20:
                img = env.render(mode=render_mode)
                img = Image.fromarray(img)
                frames.append(img.resize((img.width // 2, img.height // 2)))
            if envname == "hockey":
                ob2 = env.obs_agent_two()
            if done:
                break
        test_stats.append([i,total_reward,t+1])
        if envname == "hockey":
            wins.append(info["winner"])

    test_stats_np = np.array(test_stats)
    print("Mean test reward {} +/- std {}".format(np.mean(test_stats_np[:,1]), np.std(test_stats_np[:,1])))
    if envname == "hockey":
        print(f"{i+1} episodes completed: Fraction wins: {Counter(wins)[1]/config["numtestepisodes"]}, Fraction draws: {Counter(wins)[0]/config["numtestepisodes"]}, Fraction losses: {Counter(wins)[-1]/config["numtestepisodes"]}")

    if save_gif:
        frames[0].save("../gifs/" + filename[:-3] + "gif", save_all=True, append_images=frames[1:], duration=10, loop=0, optimize=True)
        # imageio.mimsave("../gifs/" + filename[:-3] + "gif", frames, duration=0.01)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="RL_project", description="Implements DQN and variation algorithms on various environments")

    # Test:
    parser.add_argument("--numtestepisodes", type=int, default=1000, help="Number of test episodes")
    parser.add_argument("--numteststeps", type=int, default=500, help="Number of steps per episode")
    parser.add_argument("--render", action="store_true", help="Render the environment?")
    parser.add_argument("--filename", type=str, help="Model filename to load")
    parser.add_argument("--savegif", action="store_true", help="render animated gif of agent playing")

    args = parser.parse_args()
    config = vars(args)

    saved = pk.load(open(f"../saved/{config['filename'][:-2]}k", 'rb'))
    config_train = saved["config"]
    config = {**config, **config_train}
    config["opponent"] = "strong"
    config["alpha_decay_every"] = 10
    config["alphadecay"] = 1
    config["numtestepisodes"] = 1000
    print(config)

    test_agent(config)
