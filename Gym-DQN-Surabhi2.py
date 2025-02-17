import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import torch
from feedforward import DQNAgent
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import hockey.hockey_env as h_env
import argparse


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


def train(agent, env, numepisodes, numsteps):
    episode_rewards = []
    cum_mean_episode_rewards = []
    losses = []
    printevery = numepisodes // 20

    for i in range(numepisodes):
        # print(f"Starting episode {i+1}")
        ob, _ = env.reset()
        total_reward = 0
        for _ in range(numsteps):
            a = agent.act(ob)
            (ob_new, reward, done, _, _) = env.step(a)
            total_reward += reward
            agent.store_transition((ob, a, reward, ob_new, done))
            ob = ob_new
            if done:
                break

        # print(f"Episode {i+1} ended after {t+1} steps. Episode reward = {total_reward}")

        episode_rewards.append(total_reward)
        cum_mean_episode_rewards.append(np.mean(episode_rewards[-printevery:]))
        losses.append(np.mean(agent.train(1)))

        if (i + 1) % printevery == 0:
            print(
                f"{i+1} episodes completed: Mean cumulative reward: {np.mean(episode_rewards[-printevery:])}"
            )


def train_agent(args=None):
    envname = args.env
    if envname == "hockey":
        env = h_env.HockeyEnv()
    else:
        env = gym.make(envname)
        if isinstance(env.action_space, spaces.Box):
            env = DiscreteActionWrapper(env, args.numdiscreteactions)

    o_space = env.observation_space
    a_space = env.action_space
    print(envname, o_space, a_space)

    agent = DQNAgent(
        o_space, a_space, eps=0.2, update_Qt_after=20, PrioritizedMemory=True
    )

    train(agent)

    if args.save:
        save(agent)
    if args.test:
        test(agent)
    if args.plot:
        plot(agent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="RL_project",
        description="Implements DQN and variation algorithms on various environments",
    )

    # Environment:
    parser.add_argument(
        "--env", type=str, default="hockey", help="pendulum, cartpole, or hockey"
    )
    parser.add_argument(
        "--numdiscreteactions",
        type=int,
        default=8,
        help="For continuous action spaces, the number of actions to discretize. Ignored for discrete environments.",
    )

    # Algorithm:
    parser.add_argument(
        "--double", action="store_true", help="Use Double DQN? (default: False)"
    )
    parser.add_argument(
        "--per",
        action="store_true",
        help="Use Prioritized Experience Replay? (default: False)",
    )
    parser.add_argument(
        "--dueling", action="store_true", help="Use Dueling Network? (default: False)"
    )
    parser.add_argument(
        "--rnd",
        action="store_true",
        help="Use Random Network Distillation? (default: False)",
    )

    # Handling multistep correctly:
    parser.add_argument(
        "--multistep",
        type=str,
        default="None",
        help='Multistep learning: None (1-step), int (n-step), or "MonteCarlo".',
    )

    # Hyperparameters:
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--alpha", type=float, default=0.002, help="Learning rate")
    parser.add_argument(
        "--epsilon", type=float, default=0.5, help="Epsilon for epsilon greedy"
    )
    parser.add_argument(
        "--decayepsilon",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Epsilon decay mode: 0 (no decay), 1 (linear), 2 (exponential)",
    )
    parser.add_argument(
        "--minepsilon",
        type=float,
        default=0.005,
        help="if epsilon decay mode is 1 (linear) or 2 (exponential), what min value to decay to",
    )

    # Memory:
    parser.add_argument(
        "--buffersize", type=int, default=int(1e5), help="memory buffer size"
    )
    parser.add_argument(
        "--batchsize", type=int, default=128, help="sampling batch size"
    )

    # Train:
    parser.add_argument(
        "--numepisodes", type=int, default=10000, help="number of train episodes"
    )
    parser.add_argument(
        "--numsteps", type=int, default=500, help="number of steps per episode"
    )

    # Supp:
    parser.add_argument(
        "--save", action="store_true", default=True, help="saves model (default: True)"
    )
    parser.add_argument(
        "--no-save", action="store_false", dest="save", help="don't save model"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        default=True,
        help="evaluates trained model (default: True)",
    )
    parser.add_argument(
        "--no-test", action="store_false", dest="test", help="don't evaluate model"
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="plots eval performance (default: True)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot",
        help="don't plot eval performance",
    )

    # parser.add_argument(
    #     "--verbose",
    #     action="store_true",
    #     help="verbose prints? (default: False)",
    # )

    args = parser.parse_args()

    if args.multistep.lower() == "none":
        args.multistep = None
    elif args.multistep.lower() == "montecarlo":
        args.multistep = "MonteCarlo"
    elif args.multistep.isdigit():
        args.multistep = int(args.multistep)
    else:
        raise ValueError(f"Invalid --multistep value: {args.multistep}")

    print(args)

    # train_agent(args=args)
