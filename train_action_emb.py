import json
import torch

from src.buffer import ReplayMemory, ExperienceDataset
from src.action_embeddings import EmbeddingModel, Forward, ActionEmbTrainer
from src.nn_utils import ModelLoader
from src.siameze import Encoder

f = open('envs.json')
json_config = json.load(f)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Dataset
buffer = ReplayMemory(capacity=None,
                      device=device,
                      **json_config['sampling'])
dt_iter = ExperienceDataset(buffer)

# Load All Models for Trainer
enc = Encoder(h=json_config['h_frame'], w=json_config['w_frame'], enc_size=json_config['enc_size'])
encoder = ModelLoader(path='models/encoder.pt',
                      model_to_load=enc,
                      frozen=True)

embedding = EmbeddingModel(num_embeddings=json_config['n_actions'],
                           embedding_dim=json_config['emb_depth']).to(device)
forward = Forward().to(device)

# Initialize Trainer
trainer = ActionEmbTrainer(encoder, embedding, forward, num_l=18, tensorboard=False)


for epoch_indx in range(2):
    dt_iter = ExperienceDataset(buffer)
    last_loss = trainer.train_one_epoch(epoch_indx, dt_iter, printing_batch = 1)

embedding.store('actions_embedding.pt')