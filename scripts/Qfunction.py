import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from feedforward import Feedforward, Feedforward_Dueling
import torch.optim.lr_scheduler as lr_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QFunction(Feedforward):
    def __init__(self, observation_dim, action_dim, config):
        super().__init__(input_size=observation_dim, hidden_size=config["hiddensize"], output_size=action_dim, activation=config["activation"])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["alpha"], eps=0.000001)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=config["alpha_decay_every"], gamma=config["alphadecay"])
        self.loss = torch.nn.SmoothL1Loss(reduction="none")

    def fit(self, Qval, targets, weights):
        weights = torch.tensor(weights, device=device, dtype=torch.float32)
        self.train()
        self.optimizer.zero_grad()
        td_error = torch.abs(Qval - targets)
        loss = self.loss(Qval, targets)
        loss = loss * weights
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item(), td_error

    def Q_value(self, observations, actions):
        toret = self.forward(observations).gather(1, actions)
        return toret

    def maxQ(self, observations):
        pred = self.predict(observations)
        return np.max(pred, axis=-1, keepdims=True)

    def maxQactions(self, observations):
        acts = torch.from_numpy(self.predict(observations)).argmax(dim=1, keepdim=True)
        return acts

    def doubleQt(self, observations, actions):
        toret = torch.from_numpy(self.predict(observations)).to(device).gather(1, actions)
        return toret

    def greedyAction(self, observations):
        pred = self.predict(observations)
        return np.argmax(pred, axis=-1)


class QFunction_Dueling(Feedforward_Dueling):
    def __init__(self, observation_dim, action_dim, config):
        super().__init__(input_size=observation_dim, hidden_size=config["hiddensize"], output_size=action_dim, activation=config["activation"])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["alpha"], eps=0.000001)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=config["alpha_decay_every"], gamma=config["alphadecay"])
        self.loss = torch.nn.SmoothL1Loss(reduction="none")

    def fit(self, Qval, targets, weights):
        weights = torch.tensor(weights, device=device, dtype=torch.float32)
        self.train()
        self.optimizer.zero_grad()
        td_error = torch.abs(Qval - targets)
        loss = self.loss(Qval, targets)
        loss = loss * weights
        loss = loss.mean()
        loss.backward()
        clip_grad_norm_(self.parameters(), 10.0)        # difference
        self.optimizer.step()
        self.scheduler.step()
        return loss.item(), td_error

    def Q_value(self, observations, actions):
        toret = self.forward(observations).gather(1, actions)
        return toret

    def maxQ(self, observations):
        pred = self.predict(observations)
        return np.max(pred, axis=-1, keepdims=True)

    def maxQactions(self, observations):
        acts = torch.from_numpy(self.predict(observations)).argmax(dim=1, keepdim=True)
        return acts

    def doubleQt(self, observations, actions):
        toret = torch.from_numpy(self.predict(observations)).to(device).gather(1, actions)
        return toret.cpu().numpy()

    def greedyAction(self, observations):
        pred = self.predict(observations)
        return np.argmax(pred, axis=-1)
