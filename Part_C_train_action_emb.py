import json
import torch
from matplotlib import pyplot as plt

from src.buffer import ReplayMemory, ExperienceDataset
from src.action_embeddings import EmbeddingModel, Forward, ActionEmbTrainer
from src.nn_utils import ModelLoader
from src.siameze import Encoder

# Find me also here
# https://colab.research.google.com/drive/1VcoPaXUNbKSkUEfkVFn-Qr12SaMvQQYc?usp=sharing

if __name__ == '__main__':
    f = open('envs.json')
    json_config = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Dataset
    buffer = ReplayMemory(capacity=json_config['replay_capacity'],
                          device=device,
                          **json_config['sampling'])

    # Load All Models for Trainer
    enc = Encoder(h=json_config['h_frame'],
                  w=json_config['w_frame'],
                  enc_size=json_config['enc_size']).to(device)
    # Set frozen to True so that the model is not trainable
    encoder = ModelLoader(path='models/encoder.pt',
                          model_to_load=enc,
                          frozen=True)
    # Initialize Embedding and Forward network
    embedding = EmbeddingModel(num_embeddings=json_config['n_actions'],
                               embedding_dim=json_config['emb_depth']).to(device)
    forward = Forward().to(device)

    # Initialize Trainer
    trainer = ActionEmbTrainer(encoder, embedding, forward, num_l=18, tensorboard=False)

    epochs = 2
    for epoch_indx in range(epochs):
        dt_iter = ExperienceDataset(buffer)
        loss = trainer.train_one_epoch(epoch_indx, dt_iter, printing_batch=3)

        plt.title(f'Loss History')
        plt.plot(loss)
        plt.show()

    embedding.store('models/embeddings.pt')