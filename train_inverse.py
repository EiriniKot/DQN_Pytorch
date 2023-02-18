import json, os
import torch

from src.buffer import ReplayMemory, ExperienceDataset
from src.siameze import SiamezeTrainer, Encoder, Inverse

f = open('envs.json')
json_config = json.load(f)

# You need to keep the same settings
buffer = ReplayMemory(capacity=None, **json_config['sampling'])
dt_iter = ExperienceDataset(buffer)

encoder = Encoder(h=json_config['h_frame'],
                  w=json_config['w_frame'],
                  enc_size=json_config['enc_size'])

inverse = Inverse(emb_size=json_config['enc_size'], output_size=18)
trainer = SiamezeTrainer(encoder, inverse, tensorboard=False)

models_save = '/home/eirini/PycharmProjects/DQN_Pytorch/models_inverse_encoded'

for epoch_indx in range(1):
   last_loss = trainer.train_one_epoch(epoch_indx, dt_iter)

torch.save({'epoch': epoch_indx,
            'model_state_dict': encoder.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss': last_loss}, os.path.join(models_save, f'checkpoint_{epoch_indx}_encoder.pt'))

torch.save({'epoch': epoch_indx,
           'model_state_dict': inverse.state_dict(),
           'optimizer_state_dict': trainer.optimizer.state_dict(),
           'loss': last_loss}, os.path.join(models_save, f'checkpoint_{epoch_indx}_inverse.pt'))