import torch
import numpy as np
import gymnasium as gym
from gymnasium import *
from memory import Memory, PrioritizedMemory


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])]
        )
        self.activations = [torch.nn.Tanh() for l in self.layers]
        self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)

    def forward(self, x):
        for layer, activation_fun in zip(self.layers, self.activations):
            x = activation_fun(layer(x))
        return self.readout(x)

    def predict(self, x):
        with torch.no_grad():
            return self.forward(torch.from_numpy(x.astype(np.float32))).numpy()


class QFunction(Feedforward):
    def __init__(
        self, observation_dim, action_dim, hidden_sizes=[100, 100], learning_rate=0.0002
    ):
        super().__init__(
            input_size=observation_dim,
            hidden_sizes=hidden_sizes,
            output_size=action_dim,
        )
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, eps=0.000001
        )
        self.loss = torch.nn.SmoothL1Loss(reduction="none")

    def fit(self, Qval, targets, weights):
        weights = torch.tensor(weights, device=device, dtype=torch.float32)
        self.train()  # put model in training mode
        self.optimizer.zero_grad()
        loss = self.loss(Qval, targets)
        loss = loss * weights
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def Q_value(self, observations, actions):
        return self.forward(observations).gather(1, actions)

    def maxQ(self, observations):
        pred = self.predict(observations)
        return np.max(pred, axis=-1, keepdims=True)
        # keepdims for matrix multiplication later

    def greedyAction(self, observations):
        pred = self.predict(observations)
        return np.argmax(pred, axis=-1)
        # do not actually need axis = -1 as pred will be 1D


class DQNAgent(object):
    def __init__(self, observation_space, action_space, **userconfig):

        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace(
                "Observation space {} incompatible "
                "with {}. (Require: Box)".format(observation_space, self)
            )
        if not isinstance(action_space, spaces.discrete.Discrete):
            raise UnsupportedSpace(
                "Action space {} incompatible with {}."
                " (Reqire Discrete.)".format(action_space, self)
            )

        self._observation_space = observation_space
        self._action_space = action_space
        self._action_n = action_space.n
        self.train_iter = 1
        self._config = {
            "eps": 0.05,  # Epsilon in epsilon greedy policies
            "discount": 0.95,
            "buffer_size": int(1e5),
            "batch_size": 128,
            "learning_rate": 0.0002,
            "update_Qt_after": 10,
            "PrioritizedMemory": False,
        }
        self._config.update(userconfig)
        self._eps = self._config["eps"]
        if self._config["PrioritizedMemory"]:
            self.buffer = PrioritizedMemory(max_size=self._config["buffer_size"])
        else:
            self.buffer = Memory(max_size=self._config["buffer_size"])

        self.Q = QFunction(
            self._observation_space.shape[0],
            self._action_n,
            learning_rate=self._config["learning_rate"],
        )
        self.Qt = QFunction(
            self._observation_space.shape[0],
            self._action_n,
            learning_rate=0,
        )

    def _update_target_net(self):
        self.Qt.load_state_dict(self.Q.state_dict())

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
        if np.random.random() > eps:
            action = self.Q.greedyAction(observation)
        else:
            action = self._action_space.sample()
        return action

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def train(self, iter_fit=32):
        self.train_iter += 1
        if self.train_iter % self._config["update_Qt_after"] == 0:
            self._update_target_net()
        losses = []
        for i in range(iter_fit):
            if self._config["PrioritizedMemory"]:
                sample, weights, inds = self.buffer.sample(self._config["batch_size"])
            else:
                sample = self.buffer.sample(self._config["batch_size"])
                weights = np.ones((sample.shape[0], 1))

            s = np.stack(sample[:, 0])  # s_t (batchsize,3)
            a = np.stack(sample[:, 1])[:, None]  # a_t (batchsize,1)
            rew = np.stack(sample[:, 2])[:, None]  # rew  (batchsize,1)
            s_ = np.stack(sample[:, 3])  # s_t+1 (batchsize,3)
            done = np.stack(sample[:, 4])[:, None]  # done signal  (batchsize,1)

            maxQtval = self.Qt.maxQ(s_)
            targets = rew + (1 - done) * self._config["discount"] * maxQtval
            targets = torch.tensor(targets, device=device, dtype=torch.float32)
            # print("TARGETS SHAPE", targets.shape)

            Qvals = self.Q.Q_value(
                torch.tensor(s, device=device, dtype=torch.float32),
                torch.tensor(a, device=device),
            )

            fit_loss = self.Q.fit(Qvals, targets, weights)
            losses.append(fit_loss)

            if self._config["PrioritizedMemory"]:
                sample, weights, inds = self.buffer.sample(self._config["batch_size"])
                self.buffer.update(inds)
        return losses
