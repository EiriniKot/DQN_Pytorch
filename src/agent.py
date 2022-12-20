from src.network import DqnNet
import random
import math
import torch
from torch import nn
import torch.optim as optim


class DqnAgent:
    def __init__(self, network_obj,
                 n_actions,
                 device,
                 optimizer='RMSprop',
                 target_freq = 15,
                 gamma=0.99,
                 eps_start=0.9,
                 eps_end=0.05,
                 eps_decay=200):
        assert isinstance(network_obj, DqnNet), "Should pass a instance of a DqnNet"
        self.device = device

        self.policy_net = network_obj.to(device)
        self.target_net = network_obj.to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.target_freq = target_freq
        # Disables grad eval
        self.target_net.eval()

        self.optimizer = getattr(optim, optimizer)(self.policy_net.parameters())

        self.gamma = gamma
        self.n_actions = n_actions

        # Parameters for epsilon greedy policy
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.eps_decay = eps_decay

        self.steps_done = 0

        self.loss_saver = []
        # Compute Huber loss
        self.criterion = nn.SmoothL1Loss()

    def policy(self, state):
        """
        This function takes s input argument a state and returns the index of an action.
        This actions is selected either greedely or randomly
        :param state: torch.Tensor
        :return: torch.Tensor
        """
        prop_random = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if prop_random > eps_threshold:
            # GREEDY
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                values = self.policy_net(state)
                sample_max_val_index = values.max(1)
                sample_max_index = sample_max_val_index[1]
                batched_index = sample_max_index.view(1, 1).detach()
        else:
            batched_index = torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

        return batched_index

    def loss_fn(self, experience):
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                experience.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in experience.next_state
                                           if s is not None], 0)
        state_batch = torch.cat(experience.state, 0)
        action_batch = torch.cat(experience.action, 0)
        reward_batch = torch.cat(experience.reward, 0)
        out = self.policy_net(state_batch)
        state_action_values = out.gather(1, action_batch)
        next_state_values = torch.zeros(len(experience.state))
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.loss_saver.append(loss.detach().numpy())
        return loss

    def train(self, experience):
        # Optimize the model
        self.loss_fn(experience)
        self.optimizer.step()

    def update_target(self, t):
        # Update the target network, copying all weights and biases in DQN
        if t % self.target_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())





