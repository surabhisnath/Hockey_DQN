import torch
import numpy


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
        self.loss = torch.nn.SmoothL1Loss()  # MSELoss()

    def fit(self, observations, actions, targets):
        self.train()  # put model in training mode       # feedforward's function train()
        self.optimizer.zero_grad()
        # Forward pass
        acts = torch.from_numpy(actions)
        pred = self.Q_value(torch.from_numpy(observations).float(), acts)
        # Compute Loss
        loss = self.loss(pred, torch.from_numpy(targets).float())

        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def Q_value(self, observations, actions):
        return self.forward(observations).gather(1, actions[:, None])

    def maxQ(self, observations):
        return np.max(self.predict(observations), axis=-1, keepdims=True)

    def greedyAction(self, observations):
        return np.argmax(self.predict(observations), axis=-1)
