import json
import torch
import cv2

from matplotlib import pyplot as plt

from src.network import DqnNetAlternative
from src.nn_utils import GamesRunner
from src.nn_utils import ModelLoader

if __name__ == '__main__':
    f = open('envs.json')
    json_config = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy_net = DqnNetAlternative(h=json_config['h_frame'],
                                w=json_config['w_frame'],
                                enc_size=json_config['enc_size'],
                                emb_depth=json_config['emb_depth'],
                                n_actions=json_config['n_actions'],
                                device=device,
                                encoder_path='models/encoder.pt',
                                embed_path='models/embeddings.pt')


policy_net = ModelLoader(path='models/policy_alternative.pt',
                         model_to_load=policy_net,
                         frozen=False).model_loaded

target_net = DqnNetAlternative(h=json_config['h_frame'],
                                w=json_config['w_frame'],
                                enc_size=json_config['enc_size'],
                                emb_depth=json_config['emb_depth'],
                                n_actions=json_config['n_actions'],
                                device=device,
                                encoder_path='models/encoder.pt',
                                embed_path='models/embeddings.pt')

target_net = ModelLoader(path='models/target_alternative.pt',
                         model_to_load=target_net,
                         frozen=False).model_loaded

runner = GamesRunner(json_config,
                    batch=50,
                    envs=json_config['final_env'],
                    h=json_config['h_frame'],
                    w=json_config['w_frame'],
                    tau=json_config['tau'],
                    max_iterations_ep=json_config['max_iterations_ep'],
                    capacity=100,
                    device=device,
                    p_net=policy_net,
                    t_net=target_net,
                    animation=True,
                    save_buffer=False,
                    num_episodes=15)

scores, loss = runner.run()

for game, score in scores.items():
    plt.title(f'Scores for {game}')
    plt.plot(score)
    plt.show()

plt.title(f'Loss History')
plt.plot(loss)
plt.show()
