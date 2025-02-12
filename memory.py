import numpy as np
import torch
import random


# class to store transitions
class Memory:
    def __init__(self, max_size=100000):
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size = max_size

    def add_transition(self, transitions_new):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx, :] = np.asarray(
            transitions_new, dtype=object
        )
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size
        self.inds = np.random.choice(range(self.size), size=batch, replace=False)
        return self.transitions[self.inds, :]

    def get_all_transitions(self):
        return self.transitions[0 : self.size]


class PrioritizedMemory:
    def __init__(self, max_size=100000, eps=1e-2, alpha=0.1, beta=0.1):

        self.tree = SumTree(max_size=max_size)
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size = max_size
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

    def add_transition(self, transitions_new):
        self.tree.add(self.max_priority, self.current_idx)
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx, :] = np.asarray(
            transitions_new, dtype=object
        )
        self.current_idx = (self.current_idx + 1) % self.max_size
        self.size = min(self.max_size, self.size + 1)

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch, 1, dtype=torch.float)
        segment = self.tree.total / batch
        for i in range(batch):
            a, b = segment * i, segment * (i + 1)
            cumsum = random.uniform(a, b)
            tree_idx, priority, sample_idx = self.tree.get(cumsum)
            priorities[i] = torch.tensor(priority, dtype=torch.float32)
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        probs = priorities / self.tree.total
        weights = (self.size * probs) ** -self.beta
        weights = weights / weights.max()
        return self.transitions[np.array(sample_idxs), :], weights, tree_idxs

    def get_all_transitions(self):
        return self.transitions[0 : self.size]

    def update(self, inds, priorities):
        for ind, priority in zip(inds, priorities):
            priority = (priority + self.eps) ** self.alpha
            self.tree.update(ind, priority)
            self.max_priority = max(self.max_priority, priority)


class SumTree:
    def __init__(self, max_size):
        self.nodes = [0] * (2 * max_size - 1)
        self.data = [None] * max_size

        self.max_size = max_size
        self.current_idx = 0
        self.size = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        idx = data_idx + self.max_size - 1  # child index in tree array
        change = value - self.nodes[idx]

        self.nodes[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data):

        self.data[self.current_idx] = data
        self.update(self.current_idx, value)
        self.current_idx = (self.current_idx + 1) % self.max_size
        self.size = min(self.max_size, self.size + 1)

    # def get(self, cumsum):
    #     assert cumsum <= self.total

    #     idx = 0
    #     while 2 * idx + 1 < len(self.nodes):
    #         left, right = 2 * idx + 1, 2 * idx + 2

    #         if cumsum <= self.nodes[left]:
    #             idx = left
    #         else:
    #             idx = right
    #             cumsum = cumsum - self.nodes[left]

    #     data_idx = idx - self.max_size + 1

    #     return data_idx, self.nodes[idx], self.data[data_idx]
    def get(self, cumsum):
        # print(cumsum, self.nodes[0])
        assert (
            cumsum < self.total
        ), f"cumsum {cumsum} must be strictly less than total {self.total}"

        idx = 0
        # while 2 * idx + 1 < len(self.nodes):
        while idx < self.max_size - 1:
            left, right = 2 * idx + 1, 2 * idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum -= self.nodes[left]

        data_idx = idx - self.max_size + 1

        # Ensure data_idx is within the real buffer
        if data_idx >= self.size:
            print(
                f"Warning: data_idx {data_idx} is out of range (size={self.size}). Returning random valid index."
            )
            print(cumsum, self.nodes[0], self.nodes[-1])
            data_idx = np.random.randint(0, self.size)  # Select a valid index randomly

        return data_idx, self.nodes[idx], self.data[data_idx]

    def __repr__(self):
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"


class HindsightMemory:
    def __init__(self, max_size=100000):
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size = max_size

    def add_transition(self, transitions_new):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx, :] = np.asarray(
            transitions_new, dtype=object
        )
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch=1):
        if batch > self.size:
            batch = self.size
        self.inds = np.random.choice(range(self.size), size=batch, replace=False)
        return self.transitions[self.inds, :]

    def get_all_transitions(self):
        return self.transitions[0 : self.size]
