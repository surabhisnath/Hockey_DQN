import numpy as np
import torch
import random


# class to store transitions
class Memory:
    def __init__(self, config):
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size = config["buffersize"]
        self.batchsize = config["batchsize"]
        self.multistep = config["multistep"]
        self.discount = config["gamma"]
        self.inds = []

    def add_transition(self, transitions_new):
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx, :] = np.asarray(
            transitions_new, dtype=object
        )
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self):
        if self.batchsize > self.size:
            self.batchsize = self.size
        self.inds = np.random.choice(range(self.size), size=self.batchsize, replace=False)
        if self.multistep == None:
            samples = self.transitions[self.inds, :]
        elif self.multistep == "MonteCarlo":
            samples = []
            for i in self.inds:
                sum_reward = 0
                states_look_ahead = self.transitions[i][3]
                done_look_ahead = self.transitions[i][4]
                n = 0
                while not done_look_ahead:
                    if len(self.transitions) <= i + n:
                        break
                    epstep = self.transitions[i + n][6]
                    if epstep == 0 and n > 0:
                        break
                    sum_reward += (self.discount**n) * self.transitions[i + n][2]
                    done_look_ahead = self.transitions[i + n][4]
                    n += 1
                sample = np.asarray(
                    (
                        self.transitions[i][0],
                        self.transitions[i][1],
                        sum_reward,
                        self.transitions[i][3],
                        self.transitions[i][4],
                    ),
                    dtype=object,
                )
                samples.append(sample)
        else:
            samples = []
            for i in self.inds:
                sum_reward = 0
                states_look_ahead = self.transitions[i][3]
                done_look_ahead = self.transitions[i][4]
                for n in range(self.multistep):
                    if len(self.transitions) <= i + n:
                        break
                    epstep = self.transitions[i + n][6]
                    if epstep == 0 and n > 0:
                        break
                    sum_reward += (self.discount**n) * self.transitions[i + n][2]
                    states_look_ahead = self.transitions[i + n][3]
                    done_look_ahead = self.transitions[i + n][4]
                    if done_look_ahead:
                        break
                sample = np.asarray(
                    (
                        self.transitions[i][0],
                        self.transitions[i][1],
                        sum_reward,
                        states_look_ahead,
                        done_look_ahead,
                    ),
                    dtype=object,
                )
                samples.append(sample)
        return np.asarray(samples)

    def get_all_transitions(self):
        return self.transitions[0 : self.size]


class PrioritizedMemory:
    def __init__(self, config):

        self.tree = SumTree(max_size=config["buffersize"])
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size = config["buffersize"]
        self.batchsize = config["batchsize"]
        self.multistep = config["multistep"]
        self.discount = config["gamma"]

        self.eps = 1e-2  # minimal priority, prevents zero probabilities
        self.alpha = 0.5  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = 0.5  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = 1e-2  # priority for new samples, init as eps

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

    def sample(self):
        if self.batchsize > self.size:
            self.batchsize = self.size

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(self.batchsize, 1, dtype=torch.float)
        segment = self.tree.total / self.batchsize
        for i in range(self.batchsize):
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

    def get(self, cumsum):

        assert (
            cumsum <= self.total
        ), f"cumsum {cumsum} must be strictly less than total {self.total}"

        idx = 0
        # while 2 * idx + 1 < len(self.nodes):
        index_values = []
        while idx < self.max_size - 1:
            left, right = 2 * idx + 1, 2 * idx + 2

            if cumsum <= self.nodes[left]:
                if self.nodes[left] > 0:
                    idx = left
                else:
                    break
            else:
                if self.nodes[right] > 0:
                    idx = right
                    cumsum -= self.nodes[left]
                else:
                    break
            index_values.append(float(self.nodes[idx]))

        data_idx = idx - self.max_size + 1

        # Ensure data_idx is within the real buffer
        if data_idx >= self.size:
            # print("CUMSUM:", float(cumsum), "TREEMAX:", float(self.nodes[0]))
            # print("INDEXVALUES", index_values)
            assert data_idx < self.size

        #     print(
        #         f"Warning: data_idx {data_idx} is out of range (size={self.size}). Returning random valid index."
        #     )
        #     print(cumsum, self.nodes[0])
        #     data_idx = np.random.randint(0, self.size)  # Select a valid index randomly

        return data_idx, self.nodes[idx], self.data[data_idx]

    def __repr__(self):
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"
