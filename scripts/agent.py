import torch
import numpy as np
import gymnasium as gym
from gymnasium import *
from gymnasium import spaces
from memory import Memory, PrioritizedMemory
from Qfunction import QFunction, QFunction_Dueling
from rnd import RND

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class DQNAgent(object):
    def __init__(self, observation_space, action_space, config):

        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace("Observation space {} incompatible with {}. (Require: Box)".format(observation_space, self))
        if not isinstance(action_space, spaces.discrete.Discrete):
            raise UnsupportedSpace("Action space {} incompatible with {}. (Reqire Discrete.)".format(action_space, self))
        self._observation_space = observation_space
        self._action_space = action_space
        self._action_n = action_space.n
        self.train_iter = 0 # changed to 0
        self.config = config

        if self.config["per"]: # why not use self.config in all these initialisations?
            self.buffer = PrioritizedMemory(config)
        else:
            self.buffer = Memory(config)

        if self.config["dueling"]:
            self.Q = QFunction_Dueling(self._observation_space.shape[0], self._action_n, config).to(device)
            self.Qt = QFunction_Dueling(self._observation_space.shape[0], self._action_n, {**config, "alpha":0}).to(device)        # same config except alpha is 0
        else:
            self.Q = QFunction(self._observation_space.shape[0], self._action_n, config).to(device)
            self.Qt = QFunction(self._observation_space.shape[0], self._action_n, {**config, "alpha":0}).to(device)

        if self.config["rnd"]:
            self.rnd = RND(self._observation_space.shape[0], self._action_n, self.config)

        self._update_target_net() # added this to update target net at start

    def _update_target_net(self):
        self.Qt.load_state_dict(self.Q.state_dict())

    def act(self, observation, eps):
        # if self.config["use_noisy_nets"]:
        #     action = self.Q.greedyAction(observation)
        # else:

        # eps = self.config["epsilon"]

        if np.random.random() > eps:
            action = self.Q.greedyAction(observation)
        else:
            action = self._action_space.sample()
        return action

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def train(self):
        self.train_iter += 1

        if self.train_iter % self.config["update_Qt_after"] == 0:
            self._update_target_net()
        losses = []
        for _ in range(self.config["fititerations"]):
            if self.config["per"]:
                sample, weights, inds = self.buffer.sample()
            else:
                sample = self.buffer.sample()
                weights = np.ones((sample.shape[0], 1))

            s = np.stack(sample[:, 0])                  # s_t (batchsize,3)
            a = np.stack(sample[:, 1])[:, None]         # a_t (batchsize,1)
            rew = np.stack(sample[:, 2])[:, None]       # rew  (batchsize,1)
            s_ = np.stack(sample[:, 3])                 # s_t+1 (batchsize,3)
            done = np.stack(sample[:, 4])[:, None]      # done signal  (batchsize,1)

            if self.config["rnd"]:
                rew_i = self.rnd.intrinsic_reward(torch.from_numpy(s.astype(np.float32)))
                self.rnd.update_pred(rew_i)  # update predictor    

            if self.config["double"]:
                actions_to_use = self.Q.maxQactions(s_)
                Qtval = self.Qt.doubleQt(s_, torch.tensor(actions_to_use, device=device)).cpu().numpy()
            else:
                Qtval = self.Qt.maxQ(s_)

            if self.config["multistep"] == None:
                targets = rew + (1 - done) * self.config["gamma"] * Qtval
            elif self.config["multistep"] == "MonteCarlo":
                targets = rew
            else:
                targets = (rew + (1 - done) * (self.config["gamma"] ** self.config["multistep"]) * Qtval)
            targets = torch.tensor(targets, device=device, dtype=torch.float32)
            Qvals = self.Q.Q_value(torch.tensor(s, device=device, dtype=torch.float32), 
                                   torch.tensor(a, device=device),)

            fit_loss, td_error = self.Q.fit(Qvals, targets, weights)
            losses.append(fit_loss)

            if self.config["per"]:
                self.buffer.update(inds, td_error.detach().cpu().numpy())
        
        # decay epsilon
        # self.config["epsilon"] = self.config["epsilon"] * self.config["epsilondecay"]
        # if self.config["epsilon"] < self.config["minepsilon"]:
        #    self.config["epsilon"] = self.config["minepsilon"]
            
        return losses
