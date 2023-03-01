import json
import torch
from matplotlib import pyplot as plt

from src.network import DqnNet
from src.nn_utils import GamesRunner

# Find me also here
# https://colab.research.google.com/drive/1UEcce_d-FFU_8bT94pSJvsn-H7kHrxZz?usp=sharing

if __name__ == '__main__':
    f = open('envs.json')
    json_config = json.load(f)
    # Initialize networks
    policy_net = DqnNet(h=json_config['h_frame'], w=json_config['w_frame'], outputs=json_config['n_actions'])
    target_net = DqnNet(h=json_config['h_frame'], w=json_config['w_frame'], outputs=json_config['n_actions'])
    target_net.load_state_dict(policy_net.state_dict())

    runner = GamesRunner(json_config,
                         batch=json_config['batch_size'],
                         envs=json_config['train_envs'],
                         h=json_config['h_frame'],
                         w=json_config['w_frame'],
                         tau=json_config['tau'],
                         max_iterations_ep=json_config['max_iterations_ep'],
                         capacity=json_config['replay_capacity'],
                         device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                         p_net=policy_net,
                         t_net=target_net,
                         save_buffer=True,
                         num_episodes=100)

    scores, loss = runner.run()

    for game, score in scores.items():
        plt.title(f'Scores for {game}')
        plt.plot(score)
        plt.show()

    plt.title(f'Loss History')
    plt.plot(loss)
    plt.show()