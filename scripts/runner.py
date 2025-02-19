import gymnasium as gym
from gymnasium import spaces
from gymnasium import *
import numpy as np
import time
import torch
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import hockey.hockey_env as h_env
import argparse
from collections import Counter
from agent import DQNAgent
import pickle
from matplotlib import animation
import json
from random import randint

def random_number(n=8):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


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


def train_agent(config):
    envname = config["env"]
    if envname == "hockey":
        env = h_env.HockeyEnv()
        env.discretize_actions(config["numdiscreteactions"])
    else:
        env = gym.make(envname)
        if isinstance(env.action_space, spaces.Box):
            env = DiscreteActionWrapper(env, config["numdiscreteactions"])


    agent = DQNAgent(env.observation_space, env.action_space, config)
    if envname == "hockey" and config["opponent"] == "weak":
        opponent = h_env.BasicOpponent(weak=True)
    if envname == "hockey" and config["opponent"] == "strong":
        opponent = h_env.BasicOpponent(weak=False)
    if envname == "hockey" and config["opponent"] == "self":
        opponent = agent

    episode_rewards = []
    episode_wins = []
    cum_mean_episode_rewards = []
    losses = []
    if config["numepisodes"] >= 20:
        printevery = config["numepisodes"] // 20
    else:
        printevery = 1

    for i in range(config["numepisodes"]):
        if config["verbose"]:
            print(f"Starting episode {i+1}")
        ob, _ = env.reset()
        if envname == "hockey":
            ob2 = env.obs_agent_two()
        total_reward = 0
        for t in range(config["numsteps"]):
            a = agent.act(ob)

            if envname == "hockey":
                a1 = env.action(a)
                if config["opponent"] == "random":
                    a2 = np.random.uniform(-1,1,4)
                else:
                    a2 = opponent.act(ob2)
                    # a2 = env.action(b)                                              # @Sahiti: DO WE NEED THIS? Or would it be a2 = opponent.act(ob2)
                (ob_new, reward, done, _, info) = env.step(np.hstack([a1,a2]))
                episode_wins.append(info["winner"])
            else:
                (ob_new, reward, done, _, _) = env.step(a)
            
            total_reward += reward
            agent.store_transition((ob, a, reward, ob_new, done))
            
            ob = ob_new
            ob2 = env.obs_agent_two()
            
            if done:
                break
        
        if config["verbose"]:
            print(f"Episode {i+1} ended after {t+1} steps. Episode reward = {total_reward}")

        episode_rewards.append(total_reward)
        cum_mean_episode_rewards.append(np.mean(episode_rewards[-printevery:]))
        losses.append(np.mean(agent.train()))

        if (i + 1) % printevery == 0:
            print(f"{i+1} episodes completed: Mean cumulative reward: {np.mean(episode_rewards[-printevery:])}")
            if envname == "hockey":
                print(f"{i+1} episodes completed: Fraction wins: {Counter(episode_wins[-printevery:])[1]/printevery}, Fraction draws: {Counter(episode_wins[-printevery:])[0]/printevery}, Fraction losses: {Counter(episode_wins[-printevery:])[-1]/printevery}")

    if config["save"]:
        save_dict = {
            "Q_state_dict": agent.Q.state_dict(),
            "Qt_state_dict": agent.Qt.state_dict(),
            "config": agent.config,
            "episode_rewards": episode_rewards,
            "cum_mean_episode_rewards": cum_mean_episode_rewards,
            "losses": losses
        }
        
        with open(config["savepath"] + f"agent_{random_number()}.json", "w") as f:
            json.dump(save_dict, f)

    if config["test"]:

        if envname == "hockey":
            env = h_env.HockeyEnv()
            env.discretize_actions(config["numdiscreteactions"])
        else:
            env = gym.make(envname, render_mode="human")
            if isinstance(env.action_space, spaces.Box):
                env = DiscreteActionWrapper(env, config["numdiscreteactions"])

        frames = []
        for i in range(1):
            ob, _ = env.reset()
            if envname == "hockey":
                ob2 = env.obs_agent_two()
            
            for _ in range(250):
                frames.append(env.render(mode="rgb_array"))     # uncomment to save gif
                # env.render()
                done = False        
                a = agent.act(ob)
                if envname == "hockey":
                    a1 = env.action(a)
                    a2 = opponent.act(ob2)
                    (ob_new, reward, done, _, _) = env.step(np.hstack([a1,a2]))
                else:
                    (ob_new, reward, done, _, _) = env.step(a)
                ob=ob_new
                if envname == "hockey":
                    ob2 = env.obs_agent_two()
                if done:
                    break

        env.close()
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
        patch = plt.imshow(frames[0])
        plt.axis('off')
        def animate(i):
            patch.set_data(frames[i])
        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=0.1)
        anim.save(config["savepath"] + 'gif.gif', writer='pillow', fps=500)


    if config["plot"]:
        plt.figure()
        plt.plot(cum_mean_episode_rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Mean return across episodes")
        plt.savefig(config["plotpath"] + f"cum_mean_episode_rewards.png")

        plt.figure()
        plt.plot(running_mean(episode_rewards,100))
        plt.xlabel("Training Episodes")
        plt.ylabel("Episode Return")
        plt.savefig(config["plotpath"] + f"episode_rewards.png")

        plt.figure()
        plt.plot(losses, alpha=0.3)
        plt.savefig(config["plotpath"] + f"losses.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="RL_project", description="Implements DQN and variation algorithms on various environments")

    # Environment:
    parser.add_argument("--env", type=str, default="CartPole-v0", help="pendulum, cartpole, or hockey")
    parser.add_argument("--numdiscreteactions", type=int, default=8, help="For continuous action spaces, the number of actions to discretize. Ignored for discrete environments.")

    # Algorithm:
    parser.add_argument("--double", action="store_true", help="Use Double DQN? (default: False)")
    parser.add_argument("--per", action="store_true", help="Use Prioritized Experience Replay? (default: False)")
    parser.add_argument("--dueling", action="store_true", help="Use Dueling Network? (default: False)")
    parser.add_argument("--rnd", action="store_true", help="Use Random Network Distillation? (default: False)")
    parser.add_argument("--multistep", type=str, default="None", help='Multistep learning: None (1-step), int (n-step), or "MonteCarlo".')      # cannot go with PER

    # Hyperparameters:
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--alpha", type=float, default=0.002, help="Learning rate")
    parser.add_argument("--epsilon", type=float, default=1, help="Epsilon for epsilon greedy")
    parser.add_argument("--epsilondecay", type=float, default=0.98, help="Decay factor. If 1, no decay")
    parser.add_argument("--minepsilon", type=float, default=0.001, help="Minimum value of epsilon")

    # Memory:
    parser.add_argument("--buffersize", type=int, default=int(1e5), help="Memory buffer size")
    parser.add_argument("--batchsize", type=int, default=128, help="Sampling batch size")

    # Train:
    parser.add_argument("--hiddensize", type=int, default=128, help="Hidden layer dimensionality")
    parser.add_argument("--activation", default="ReLU", help="Activation function to use")
    parser.add_argument("--numseeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--numepisodes", type=int, default=20000, help="Number of train episodes")
    parser.add_argument("--numsteps", type=int, default=500, help="Number of steps per episode")
    parser.add_argument("--fititerations", type=int, default=32, help="Number of fit iterations per episode")
    parser.add_argument("--update_Qt_after", type=int, default=1000, help="Update target network after every")

    # Hockey:
    parser.add_argument("--opponent", default="weak", help="random/weak/strong/self opponent")
    parser.add_argument("--curriculum", action="store_true", help="Use curriculum learning (train shoot then defense then combination)? (default: False)")

    # Supp:
    parser.add_argument("--save", action="store_true", default=True, help="Saves model (default: True)")
    parser.add_argument("--nosave", action="store_false", dest="save", help="Don't save model")
    parser.add_argument("--savepath", default="../saved/", help="Path to save model, unless --nosave")

    parser.add_argument("--test", action="store_true", default=True, help="Evaluates trained model (default: True)")
    parser.add_argument("--notest", action="store_false", dest="test", help="Don't evaluate model")

    parser.add_argument("--plot", action="store_true", default=True, help="Plots eval performance (default: True)")
    parser.add_argument("--noplot", action="store_false", dest="plot", help="Don't plot eval performance")
    parser.add_argument("--plotpath", default="../plots/", help="Path to save plots, unless --noplot")

    parser.add_argument("--verbose", action="store_true", help="Verbose prints? (default: False)")

    args = parser.parse_args()

    if args.multistep.lower() == "none":
        args.multistep = None
    elif args.multistep.lower() == "montecarlo":
        args.multistep = "MonteCarlo"
    elif args.multistep.isdigit():
        args.multistep = int(args.multistep)
    else:
        raise ValueError(f"Invalid --multistep value: {args.multistep}")
    

    config = vars(args)
    print(config)

    train_agent(config)