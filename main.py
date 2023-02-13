import json
import gym
import numpy as np
import pandas as pd
import torch
from src.agent import DqnAgent
from src.network import DqnNet
from src.buffer import ReplayMemory, Transition
from itertools import count
from matplotlib import pyplot as plt
import psutil
import time
import os


class GamesRunner:
    def __init__(self, specs, h, w,
                 ram_thres=0.8,
                 batch=10,
                 capacity=100,
                 save = False,
                 num_episodes=10):

        self.envs = {}
        for env in specs['train_envs']:
            self.envs[env] = gym.make(env, render_mode='rgb_array')
        n_actions = self.envs[env].action_space.n

        self.h = h
        self.w = w
        self.batch =batch
        self.ram_thres = ram_thres
        self.save = save

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.window_size = specs['sampling']['window_size']

        self.num_episodes = num_episodes
        self.r_buffer = ReplayMemory(self.capacity,
                                     window_size=specs['sampling']['window_size'],
                                     window_step=specs['sampling']['window_step'])

        self.network_obj = DqnNet(self.h, self.w, n_actions)

        self.agent = DqnAgent(self.network_obj,
                              n_actions=n_actions,
                              target_freq=specs['target_freq'],
                              device=self.device,
                              optimizer=specs['optimizer'], **specs['policy_specs'])
        self.run_time = str(time.time())
        os.makedirs('saved_games', exist_ok=True)

    def get_init_state(self, env):
        # Initialize the environment and state
        init_state = env.reset()[0]
        init_state = np.divide(init_state, 255.)
        # BCHW
        init_state = np.transpose(init_state, (2, 0, 1))
        new_shape = (1, 3, 1, self.h, self.w)
        init_state = np.resize(init_state, new_shape)
        init_state = torch.from_numpy(init_state).type(torch.float32)

        # In the first batch we will create 3 empty frames. Since we dont have other info
        empty_states = np.full(shape=(1, 3, 3, self.h, self.w), fill_value=0.)
        empty_states = torch.tensor(empty_states).type(torch.float32)

        cat_states = torch.cat((empty_states, init_state), 2)
        del init_state, new_shape, empty_states
        return cat_states

    def run(self):
        str_info = []
        new_shape = (1, 3, 1, self.h, self.w)
        scores = {}
        for env_n, env in self.envs.items():
            print(f'Environment --- {env_n} ---')
            state = self.get_init_state(env)
            scores[env_n] = []
            self.agent.steps_done = 0

            for ep in range(self.num_episodes):
                print(f'Episode number --- {ep} ---')
                env.reset()
                sum_reward = 0

                for t in count():
                    action = self.agent.policy(state)
                    next_state, reward, done, truncated, info = env.step(action)
                    sum_reward += reward
                    # Preprocess
                    next_state = np.divide(next_state, 255.)
                    next_state = np.transpose(next_state, (2, 0, 1))
                    next_state = np.resize(next_state, new_shape)
                    next_state = torch.from_numpy(next_state).type(torch.float32)

                    # Concat 3 previous + 1 new frame
                    next_state = torch.cat((state[:, :, 1:, :, :], next_state), 2)

                    reward = torch.tensor(reward, device=None)
                    action = torch.tensor(action, device=None)

                    # Store the transition in memory
                    self.r_buffer.push(state, action, next_state, reward)

                    state = next_state.clone()

                    if len(self.r_buffer) >= self.batch and t%self.batch==0:
                        transitions = self.r_buffer.sample(self.batch)
                        experience = Transition(*zip(*transitions))
                        # Perform one step of the optimization (on the policy network)
                        self.agent.train(experience)
                        del experience, transitions, next_state, reward, action

                    self.agent.update_target(t)
                    if done:
                        print('Episode ended \n')
                        break

                    # Getting % usage of virtual_memory (3rd field)
                    ram_percentage = psutil.virtual_memory()[2]
                    print('Ram percentage over threshold', ram_percentage)
                    if ram_percentage > self.ram_thres and self.save:
                        self.r_buffer.save_local(f'saved_games/{ep}_{t}_{env_n}.pt')
                    if self.save!=True:
                        self.r_buffer.memory.clear()

                scores[env_n].append(sum_reward)
                print(sum_reward)
                str_info.append([env_n, ep, t])

            print('Final Save')
            if self.save:
                self.r_buffer.save_local(f'saved_games/{ep}_{t}_{env_n}.pt')
            plt.plot(self.agent.loss_saver)
            plt.show()

        metadata = pd.DataFrame(str_info,
                                columns=['env_name', 'episode', 'number_of_steps'])
        metadata.to_csv('metadata.csv')
        return scores


if __name__ == '__main__':
    f = open('envs.json')
    json_config = json.load(f)
    runner = GamesRunner(json_config,
                         ram_thres=70.,
                         batch=6,
                         h=json_config['h_frame'], w=json_config['w_frame'],
                         capacity=None,
                         save=True,
                         num_episodes=100)
    scores = runner.run()

    for game, score in scores.items():
        plt.plot(score)
        plt.show()







