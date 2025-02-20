import gymnasium as gym
from gymnasium import spaces
from gymnasium import *
import numpy as np
import time
import torch
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys
import os
sys.path.append(os.path.abspath("../"))
import hockey.hockey_env as h_env
import argparse
from collections import Counter
from agent import DQNAgent
import pickle as pk
from matplotlib import animation
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

    episode_rewards_seeds = []
    episode_wins_seeds = []
    cum_mean_episode_rewards_seeds = []
    losses_seeds = []
    if config["rnd"]:
        episode_intrinsic_rewards = []
    if config["numepisodes"] >= 20:
        numprints = config["numepisodes"] // 20
    else:
        numprints = 1
    
    best_agent = None
    best_agent_episode_rewards = []
    best_agent_episode_wins = []
    best_agent_cum_mean_episode_rewards = []
    best_agent_losses = []

    best_agent_eval_perf = -float("inf")
    best_agent_seed = None

    for seed in range(config["numseeds"]):
        seed = seed * 100

        agent = DQNAgent(env.observation_space, env.action_space, config)
        
        if envname == "hockey" and config["opponent"] == "weak":
            opponent = h_env.BasicOpponent(weak=True)
        if envname == "hockey" and config["opponent"] == "strong":
            opponent = h_env.BasicOpponent(weak=False)
        if envname == "hockey" and config["opponent"] == "self":
            opponent = agent

        print(f"Starting seed {seed+1}")
        np.random.seed(seed)
        torch.manual_seed(seed)

        episode_rewards = []
        episode_wins = []
        cum_mean_episode_rewards = []
        losses = []
        if config["rnd"]:
            episode_intrinsic_rewards = []


        for i in range(config["numepisodes"]):
            if config["verbose"]:
                print(f"Seed: {seed}. Starting episode {i+1}")
            ob, info = env.reset()
            if envname == "hockey":
                ob2 = env.obs_agent_two()
            total_reward = 0
            if config["rnd"]:
                total_intrinsic_reward = 0
            for t in range(config["numsteps"]):
                a = agent.act(ob)

                if envname == "hockey":
                    a1 = env.action(a)
                    if config["opponent"] == "random":
                        a2 = np.random.uniform(-1,1,4)
                    else:
                        a2 = opponent.act(ob2)
                    (ob_new, reward, done, _, info) = env.step(np.hstack([a1,a2]))
                else:
                    (ob_new, reward, done, _, _) = env.step(a)
            
                total_reward += reward

                if config["rnd"]:
                    # get intrinsic rewards
                    reward_i = agent.rnd.intrinsic_reward(
                        torch.from_numpy(ob.astype(np.float32))).detach().clamp(-1.0, 1.0).item()

                    # find combined reward
                    combined_reward = reward + reward_i
                    total_intrinsic_reward += reward_i
                    agent.store_transition((ob, a, combined_reward, ob_new, done))

                else:
                    agent.store_transition((ob, a, reward, ob_new, done))
            
                ob = ob_new
                if envname == "hockey":
                    ob2 = env.obs_agent_two()
            
                if done:
                    break

            if envname == "hockey":    
                episode_wins.append(info["winner"])
        
            if config["verbose"]:
                print(f"Seed: {seed}. Episode {i+1} ended after {t+1} steps. Episode reward = {total_reward}")

            episode_rewards.append(total_reward)
            if config["rnd"]:
                episode_intrinsic_rewards.append(total_intrinsic_reward)
            cum_mean_episode_rewards.append(np.mean(episode_rewards[-numprints:]))
            losses.append(np.mean(agent.train()))

            if (i + 1) % numprints == 0:
                print(f"Seed: {seed}. {i+1} episodes completed: Mean cumulative reward: {np.mean(episode_rewards[-numprints:])}")
                if envname == "hockey":
                    print(f"Seed: {seed}. {i+1} episodes completed: Fraction wins: {Counter(episode_wins[-numprints:])[1]/numprints}, Fraction draws: {Counter(episode_wins[-numprints:])[0]/numprints}, Fraction losses: {Counter(episode_wins[-numprints:])[-1]/numprints}")
        
        episode_rewards_seeds.append(episode_rewards)
        episode_wins_seeds.append(episode_wins)
        cum_mean_episode_rewards_seeds.append(cum_mean_episode_rewards)
        losses_seeds.append(losses)

        if envname == "hockey":
            eval_perf = test_agent(config, agent, opponent)
        else:
            eval_perf = test_agent(config, agent)
        if eval_perf > best_agent_eval_perf:
            best_agent, best_agent_episode_rewards, best_agent_episode_wins, best_agent_cum_mean_episode_rewards, best_agent_losses, best_agent_eval_perf, best_agent_seed = agent, episode_rewards, episode_wins, cum_mean_episode_rewards, losses, eval_perf, seed
    
    episode_rewards_means = np.mean(np.array(episode_rewards_seeds), axis=0)
    assert len(episode_rewards_means) == config["numepisodes"]
    episode_wins_means = np.mean(np.array(episode_wins_seeds), axis=0)

    print("Mean across seeds")
    for i in range(config["numepisodes"]):
        if (i + 1) % numprints == 0:
            print(f"{i+1} episodes completed: Mean cumulative reward: {np.mean(episode_rewards_means[i+1-numprints:i+1])}")
            if envname == "hockey":
                print(f"{i+1} episodes completed: Fraction wins: {Counter(episode_wins_means[i+1-numprints:i+1])[1]/numprints}, Fraction draws: {Counter(episode_wins_means[i+1-numprints:i+1])[0]/numprints}, Fraction losses: {Counter(episode_wins_means[i+1-numprints:i+1])[-1]/numprints}")
    
    print("Best seed")
    for i in range(config["numepisodes"]):
        if (i + 1) % numprints == 0:
            print(f"{i+1} episodes completed: Mean cumulative reward: {np.mean(best_agent_episode_rewards[i+1-numprints:i+1])}")
            if envname == "hockey":
                print(f"{i+1} episodes completed: Fraction wins: {Counter(best_agent_episode_wins[i+1-numprints:i+1])[1]/numprints}, Fraction draws: {Counter(best_agent_episode_wins[i+1-numprints:i+1])[0]/numprints}, Fraction losses: {Counter(best_agent_episode_wins[i+1-numprints:i+1])[-1]/numprints}")
    
    if envname == "hockey":
        return best_agent, best_agent_episode_rewards, best_agent_episode_wins, best_agent_cum_mean_episode_rewards, best_agent_losses, best_agent_eval_perf, best_agent_seed, opponent
    return best_agent, best_agent_episode_rewards, best_agent_episode_wins, best_agent_cum_mean_episode_rewards, best_agent_losses, best_agent_eval_perf, best_agent_seed

def test_agent(config, agent=None, opponent=None, filename=None):
    envname = config["env"]
    if envname == "hockey":
        env = h_env.HockeyEnv()
        env.discretize_actions(config["numdiscreteactions"])
    else:
        env = gym.make(envname) #add render_mode="human" for rendering
        if isinstance(env.action_space, spaces.Box):
            env = DiscreteActionWrapper(env, config["numdiscreteactions"])
    
    # TODO: load agent from file
    if not config["train"] and config["test"]:
        assert agent is None and filename is not None
        # load agent from file
        pass

    # frames = []
    test_stats = []
    for i in range(config["numtestepisodes"]):
        ob, _ = env.reset()
        if envname == "hockey":
            ob2 = env.obs_agent_two()
        total_reward = 0
        for t in range(config["numsteps"]):
            # frames.append(env.render(mode="rgb_array"))     # uncomment to save gif
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
            total_reward += reward
            if done:
                break
        test_stats.append([i, total_reward, t+1])

    test_stats_np = np.array(test_stats)
    print("Mean test reward {} +/- std {}".format(np.mean(test_stats_np[:,1]), 
                                                    np.std(test_stats_np[:,1]))) # to print test rewards
    
    # env.close()
    # plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    # patch = plt.imshow(frames[0])
    # plt.axis('off')
    # def animate(i):
    #     patch.set_data(frames[i])
    # anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=0.1)
    # anim.save(config["savepath"] + 'gif.gif', writer='pillow', fps=500)

    return np.mean(test_stats_np[:,1])

def run(config):

    if config["train"]:
        if config["env"] == "hockey":
            best_agent, best_agent_episode_rewards, best_agent_episode_wins, best_agent_cum_mean_episode_rewards, best_agent_losses, best_agent_eval_perf, best_agent_seed, opponent = train_agent(config)
        else:
            best_agent, best_agent_episode_rewards, best_agent_episode_wins, best_agent_cum_mean_episode_rewards, best_agent_losses, best_agent_eval_perf, best_agent_seed = train_agent(config)
        
    if config["train"] and config["save"]:      # save only valid when train is True. For test without train, load agent from file
        save_dict = {
            "Q_state_dict": best_agent.Q.state_dict(),
            "Qt_state_dict": best_agent.Qt.state_dict(),
            "config": config,
            "episode_rewards": best_agent_episode_rewards,
            "episode_wins": best_agent_episode_wins,
            "cum_mean_episode_rewards": best_agent_cum_mean_episode_rewards,
            "losses": best_agent_losses,
            "eval_performance": best_agent_eval_perf,
            "seed": best_agent_seed
        }
        
        ran_num = random_number()
        with open(config['savepath'] + f"agent_{config['env']}_{best_agent_seed}_{ran_num}.pk", "wb") as f:
            pk.dump(save_dict, f)

        torch.save(best_agent.Q.state_dict(),
                   f'../saved/{config["env"]}-seed{best_agent_seed}_{ran_num}.pth')

    if config["test"]:
        if config["train"]:
            if config["env"] == "hockey":
                test_agent(config, best_agent, opponent)
            else:
                test_agent(config, best_agent)
        else:
            try:
                test_agent(config, filename=config["testfile"])
            except KeyError:
                raise KeyError("Please provide a filename for the agent as --testfile ../saved/agent_(insert-number).pk") from None

    if config["train"] and config["plot"]:
        plt.figure()
        plt.plot(best_agent_cum_mean_episode_rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Mean return across episodes")
        plt.savefig(config["plotpath"] + f"agent_{ran_num}_cum_mean_episode_rewards.png")

        plt.figure()
        plt.plot(running_mean(best_agent_episode_rewards,100))
        plt.xlabel("Training Episodes")
        plt.ylabel("Episode Return")
        plt.savefig(config["plotpath"] + f"agent_{ran_num}_episode_rewards.png")

        plt.figure()
        plt.plot(best_agent_losses)
        plt.savefig(config["plotpath"] + f"agent_{ran_num}_losses.png")

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
    parser.add_argument("--alpha", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--alpha_rnd", type=float, default=0.0001, help="Learning rate for RND target network")
    parser.add_argument("--epsilon", type=float, default=0.5, help="Epsilon for epsilon greedy")
    parser.add_argument("--epsilondecay", type=float, default=0.98, help="Decay factor. If 1, no decay")
    parser.add_argument("--minepsilon", type=float, default=0.001, help="Minimum value of epsilon")

    # Memory:
    parser.add_argument("--buffersize", type=int, default=int(1e5), help="Memory buffer size")
    parser.add_argument("--batchsize", type=int, default=128, help="Sampling batch size")

    # Train:
    parser.add_argument("--train", action="store_true", default=True, help="Trains model (default: True)")
    parser.add_argument("--notrain", action="store_false", dest="train", help="Don't train model")
    parser.add_argument("--hiddensize", type=int, default=100, help="Hidden layer dimensionality")
    parser.add_argument("--activation", default="tanh", help="Activation function to use")
    parser.add_argument("--numseeds", type=int, default=2, help="Number of seeds")
    parser.add_argument("--numepisodes", type=int, default=600, help="Number of train episodes")
    parser.add_argument("--numtestepisodes", type=int, default=100, help="Number of test episodes")
    parser.add_argument("--numsteps", type=int, default=500, help="Number of steps per episode")
    parser.add_argument("--fititerations", type=int, default=32, help="Number of fit iterations per episode")
    parser.add_argument("--update_Qt_after", type=int, default=20, help="Update target network after every")

    # Hockey:
    parser.add_argument("--opponent", default="weak", help="random/weak/strong/self opponent")
    parser.add_argument("--curriculum", action="store_true", help="Use curriculum learning (train shoot then defense then combination)? (default: False)")

    # Supp:

    parser.add_argument("--save", action="store_true", default=True, help="Saves model (default: True)")
    parser.add_argument("--nosave", action="store_false", dest="save", help="Don't save model")
    parser.add_argument("--savepath", default="../saved/", help="Path to save model, unless --nosave")

    parser.add_argument("--test", action="store_true", default=True, help="Evaluates trained model (default: True)")
    parser.add_argument("--testfilename", help="Evaluates trained model (default: True)")
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

    run(config)
