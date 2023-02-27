import json
from matplotlib import pyplot as plt

from src.network import DqnNetAlternative
from src.nn_utils import GamesRunner

if __name__ == '__main__':
    f = open('envs.json')
    json_config = json.load(f)
    dqn_net = DqnNetAlternative(h=json_config['h_frame'],
                                w=json_config['w_frame'],
                                enc_size=json_config['enc_size'],
                                emb_depth=json_config['emb_depth'],
                                n_actions=json_config['n_actions'],
                                encoder_path='models_inverse_encoded/encoder.pt',
                                embed_path='models_action_emb/actions_embedding.pt')

    runner = GamesRunner(json_config,
                         batch=json_config['batch_size'],
                         envs=json_config['final_env'],
                         h=json_config['h_frame'], w=json_config['w_frame'],
                         capacity=json_config['replay_capacity'],
                         network=dqn_net,
                         save_buffer=False,
                         num_episodes=20)

    scores, loss = runner.run()

    for game, score in scores.items():
        plt.title(f'Scores for {game}')
        plt.plot(score)
        plt.show()

    plt.title(f'Loss History')
    plt.plot(loss)
    plt.show()




