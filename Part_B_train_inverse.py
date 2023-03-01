import json, os
import torch
from matplotlib import pyplot as plt

from src.buffer import ReplayMemory, ExperienceDataset
from src.siameze import SiamezeTrainer, Encoder, Inverse

# Find me also here
# https://colab.research.google.com/drive/1YH3AWF0uGNEUbK0mCwcIn2NAi05VYsFp?usp=sharing

if __name__ == '__main__':
    f = open('envs.json')
    json_config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize encoder and inverse models and place them to device
    encoder = Encoder(h=json_config['h_frame'],
                      w=json_config['w_frame'],
                      enc_size=json_config['enc_size']).to(device)
    inverse = Inverse(emb_size=json_config['enc_size'], output_size=18).to(device)

    trainer = SiamezeTrainer(encoder, inverse, tensorboard=False)

    models_save = os.path.join(os.getcwd(), 'models')
    buffer = ReplayMemory(capacity=json_config['replay_capacity'], device=device, **json_config['sampling'])
    dt_iter = ExperienceDataset(buffer, path_folder='saved_games')

    epochs = 2
    for epoch_indx in range(epochs):
      loss = trainer.train_one_epoch(epoch_indx, dt_iter, printing_batch=12)
      plt.title(f'Loss History')
      plt.plot(loss)
      plt.show()

    # Save model encoder only inverse will not be needed
    torch.save({'epoch': epoch_indx,
                'model_state_dict': encoder.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict()}, os.path.join(models_save, 'encoder.pt'))
