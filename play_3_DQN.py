import json
from matplotlib import pyplot as plt

from src.network import DqnNet
from src.nn_utils import GamesRunner

if __name__ == '__main__':
    f = open('envs.json')
    json_config = json.load(f)
    dqn_net = DqnNet(h=json_config['h_frame'], w=json_config['w_frame'], outputs=json_config['n_actions'])

    runner = GamesRunner(json_config,
                         batch=json_config['batch_size'],
                         envs = json_config['train_envs'],
                         h=json_config['h_frame'],
                         w=json_config['w_frame'],
                         max_iterations_ep=json_config['max_iterations_ep'],
                         capacity=json_config['replay_capacity'],
                         network=dqn_net,
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







