import time, os
import gym
import numpy as np
import torch
from src.agent import DqnAgent
from src.buffer import ReplayMemory, Transition
from itertools import count
import datetime
import cv2
from matplotlib import pyplot as plt

class ModelLoader:
    def __init__(self, path, model_to_load, frozen=True):
        """
        This class is created in order to load pretrained models and reuse them or/and finetune them.
        :param path: Path like .pt file
        :param model_to_load: Instance of a class for the model.
        :param frozen: bool, By frozen == True all the layers will be frozen
        and no backpropagation will be applied on them
        """
        self.model_loaded = model_to_load
        self.model_loaded.load_state_dict(torch.load(path,
                                                     map_location= torch.device('cuda' if torch.cuda.is_available() else 'cpu'))['model_state_dict'])
        if frozen:
            self.model_loaded.eval()
            for param in self.model_loaded.parameters():
                param.requires_grad = False

    def predict(self, input):
        output = self.model_loaded(input)
        return output


class GamesRunner:
    def __init__(self, specs, h, w,
                 batch=10,
                 envs=[],
                 capacity=100,
                 tau=0.001,
                 max_iterations_ep=2000,
                 save_buffer=False,
                 animation= False,
                 p_net=None,
                 t_net=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 num_episodes=10):

        self.envs = {}
        for env in envs:
            self.envs[env] = gym.make(env, render_mode='rgb_array')
        n_actions = self.envs[env].action_space.n

        self.h = h
        self.w = w
        self.tau = tau
        self.batch =batch
        self.save_buffer = save_buffer
        self.animation = animation
        if self.animation:
            os.makedirs('plots', exist_ok=True)

        self.device = device
        self.capacity = capacity
        self.window_size = specs['sampling']['window_size']

        self.num_episodes = num_episodes
        self.max_iterations_ep = max_iterations_ep

        self.r_buffer = ReplayMemory(self.capacity,
                                     window_size=specs['sampling']['window_size'],
                                     window_step=specs['sampling']['window_step'])

        self.p_network = p_net
        self.t_network = t_net

        self.agent = DqnAgent(self.p_network,
                              self.t_network,
                              n_actions=n_actions,
                              device=self.device,
                              optimizer=specs['optimizer'], **specs['policy_specs'])
        self.run_time = str(time.time())
        os.makedirs('saved_games', exist_ok=True)

    def get_init_state(self, env):
        """
        This function is responsible for building the initial state of the env considering
        the case of 4 stacked frames.
        First it resets the env and normalize frame values. Then it resizes the frames and stacks
        3 zero valued frames with the initial environment state
        :param env: gym env
        :return: cat_states: torch.Tensor
        """
        # Initialize the environment and state
        init_state = env.reset()[0]
        init_state = np.divide(init_state, 255.)
        # BCHW
        init_state = np.transpose(init_state, (2, 0, 1))
        new_shape = (1, 3, 1, self.h, self.w)
        init_state = np.resize(init_state, new_shape)
        init_state = torch.from_numpy(init_state).type(torch.float32).to(self.device)

        # In the first batch we will create 3 empty frames. Since we dont have other info
        empty_states = np.full(shape=(1, 3, 3, self.h, self.w), fill_value=0.)
        empty_states = torch.tensor(empty_states, device=self.device).type(torch.float32)

        cat_states = torch.cat((empty_states, init_state), 2)
        return cat_states

    def run(self):
        new_shape = (1, 3, 1, self.h, self.w)
        scores = {}
        for env_n, env in self.envs.items():
            print(f'Environment --- {env_n} ---')
            scores[env_n] = []
            self.agent.steps_done = 0
            for ep in range(self.num_episodes):
                print(f'Episode number --- {ep} ---')
                env.reset()
                sum_reward = 0
                state = self.get_init_state(env)

                for t in count():
                    if self.animation:
                        plt.imshow(env.render())
                        plt.grid(False)
                        plt.title(f'Episode number --- {ep} ---')
                        add_unix_time = str(int(time.mktime(datetime.datetime.now().timetuple())))
                        plt.savefig(f'plots/{ep}{add_unix_time}.png')
                        plt.close()

                    action = self.agent.policy(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    sum_reward += reward
                    # Preprocess
                    next_state = np.divide(next_state, 255.)
                    next_state = np.transpose(next_state, (2, 0, 1))
                    next_state = np.resize(next_state, new_shape)
                    next_state = torch.from_numpy(next_state).type(torch.float32).to(self.device)

                    # Concat 3 previous + 1 new frame
                    next_state = torch.cat((state[:, :, 1:, :, :], next_state), 2)
                    reward = torch.tensor(reward, device=self.device)

                    state.to(self.device)
                    action.to(self.device)
                    next_state.to(self.device)

                    # Store the transition in memory
                    self.r_buffer.push(state, action, next_state, reward)

                    del state, reward, action

                    state = next_state.clone()
                    del next_state

                    if self.r_buffer.is_full():
                        # Randomly Select some frames (=batch) to train the agent
                        transitions = self.r_buffer.sample(self.batch)
                        experience = Transition(*zip(*transitions))

                        # Perform one step of the optimization (on the policy network)
                        self.agent.train(experience, epochs=5)
                        self.agent.soft_update(self.p_network, self.t_network, tau=self.tau)

                        # Save buffer is a boolean that can switch if you want to save frames or not
                        if self.save_buffer:
                            add_unix_time = str(int(time.mktime(datetime.datetime.now().timetuple())))
                            self.r_buffer.save_local(f'saved_games/{ep}_{t}_{add_unix_time}_{env_n}.pt')
                        else:
                            self.r_buffer.memory.clear()

                        del transitions, experience

                    # Here we add an extra reason to stop the game
                    # If the game reaches max_iterations_ep without any reward
                    # we stop the game
                    reason_to_stop = t >= self.max_iterations_ep and sum_reward==0
                    if done or reason_to_stop:
                        print(f'Episode ended at {t}')
                        break

                del state
                scores[env_n].append(float(sum_reward)/t)
                print(f''
                      f'Reward  :  {sum_reward} --- and last loss  : {self.agent.loss_saver[-1]}\n')

        if self.animation:
            self.make_video(video_name='video2.avi')
        return scores, self.agent.loss_saver

    def make_video(self, video_name='video.avi'):
        image_folder = 'plots'
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter('videio.avi',
                                fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                fps=10.,
                                frameSize=(width, height))

        for image in sorted(images):
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()