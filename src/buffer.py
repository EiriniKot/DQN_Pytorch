from collections import namedtuple, deque
import random
import torch
import sys
import io
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """
    In order to collect and reuse samples while playing a game
    we need to have a Replay Buffer Object which
    """
    def __init__(self, capacity, window_size=1, window_step=1):
        self.memory = deque([], maxlen=capacity)
        self.window_size = window_size
        self.window_step = window_step

    def push(self, state, action, next_state, reward):
        """Save a transition"""
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        out = random.sample(self.memory, batch_size)
        return out

    def __len__(self):
        return len(self.memory)

    def save_local(self, output_name):
        if len(self.memory) > 0:
            torch.save(self.memory, output_name)
        self.memory.clear()

    def load_torch(self,directory):
        read_tensor = torch.load(directory)
        print('read',read_tensor)
        return read_tensor

