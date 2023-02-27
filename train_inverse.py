import json, os
import torch

from src.buffer import ReplayMemory, ExperienceDataset
from src.siameze import SiamezeTrainer, Encoder, Inverse

f = open('envs.json')
json_config = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(h=json_config['h_frame'],
                  w=json_config['w_frame'],
                  enc_size=json_config['enc_size']).to(device)

inverse = Inverse(emb_size=json_config['enc_size'], output_size=18).to(device)
trainer = SiamezeTrainer(encoder, inverse, tensorboard=False)

models_save = os.path.join(os.getcwd(), 'models')

buffer = ReplayMemory(capacity=json_config['replay_capacity'], device=device, **json_config['sampling'])
dt_iter = ExperienceDataset(buffer, path_folder='saved_games')

for epoch_indx in range(1):
  last_loss = trainer.train_one_epoch(epoch_indx, dt_iter, printing_batch=1)

torch.save({'epoch': epoch_indx,
            'model_state_dict': encoder.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss': last_loss}, os.path.join(models_save, 'encoder.pt'))
