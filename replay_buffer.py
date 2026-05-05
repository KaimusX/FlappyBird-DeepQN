# replay_buffer.py - Stores past experiences for randomized batch training.
# Sampling randomly breaks correlations between steps, stabilizing training.
# Luis

import random
from collections import deque

class ReplayBuffer:

    def __init__(self, capacity=10000):
        # Once full, oldest experiences are automatically dropped
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Store one transition: what happened, what we did, and what resulted
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Random sample without replacement
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)