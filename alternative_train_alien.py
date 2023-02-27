import json
import torch
from matplotlib import pyplot as plt

from src.network import DqnNetAlternative
from src.nn_utils import GamesRunner

if __name__ == '__main__':
    f = open('envs.json')
    json_config = json.load(f)
    policy_net = DqnNetAlternative(h=json_config['h_frame'],
                                w=json_config['w_frame'],
                                enc_size=json_config['enc_size'],
                                emb_depth=json_config['emb_depth'],
                                n_actions=json_config['n_actions'],
                                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                encoder_path='models/encoder.pt',
                                embed_path='models/embedding.pt')

    target_net = DqnNetAlternative(h=json_config['h_frame'],
                                   w=json_config['w_frame'],
                                   enc_size=json_config['enc_size'],
                                   emb_depth=json_config['emb_depth'],
                                   n_actions=json_config['n_actions'],
                                   device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                   encoder_path='models/encoder.pt',
                                   embed_path='models/embedding.pt')

    target_net.load_state_dict(policy_net.state_dict())

    runner = GamesRunner(json_config,
                         batch=json_config['batch_size'],
                         envs=json_config['final_env'],
                         h=json_config['h_frame'],
                         w=json_config['w_frame'],
                         tau=json_config['tau'],
                         max_iterations_ep=json_config['max_iterations_ep'],
                         capacity=json_config['replay_capacity'],
                         device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                         p_net=policy_net,
                         t_net=target_net,
                         save_buffer=False,
                         num_episodes=60)

    scores, loss = runner.run()

    for game, score in scores.items():
        plt.title(f'Scores for {game}')
        plt.plot(score)
        plt.show()

    plt.title(f'Loss History')
    plt.plot(loss)
    plt.show()




