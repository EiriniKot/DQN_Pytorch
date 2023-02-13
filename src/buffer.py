from collections import namedtuple, deque
import random, os
import torch
from torch.utils.data.dataset import IterableDataset

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

    def load_torch(self, directory):
        """
        Simple Function for Loading .pt tensors
        :param directory: a file-like object (has to implement :meth:`read`,
                                              :meth:`readline`, :meth:`tell`, and :meth:`seek`),
                                              or a string or os.PathLike object containing a file name
        :return: torch.tensor
        """
        read_tensor = torch.load(directory)
        return read_tensor


class ExperienceDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer) -> None:
        self.buffer = buffer
        self.path_saved = os.path.abspath('saved_games')
        self.full_paths = [os.path.join(self.path_saved, file) for file in os.listdir(self.path_saved)]

    def __iter__(self):
        for file_path in self.full_paths:
            deque_loaded = self.buffer.load_torch(file_path)
            # states, actions, rewards, dones, new_states = self.buffer.load_torch(file_path)
            for i in deque_loaded:
                yield i.state, i.action, i.reward, i.next_state



