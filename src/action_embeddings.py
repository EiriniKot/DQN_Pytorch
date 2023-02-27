import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class EmbeddingModel(nn.Module):
    def __init__(self, num_embeddings=18, embedding_dim=32):
        super(EmbeddingModel, self).__init__()
        self.emb = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    def forward(self, action):
        emb_action = self.emb(action)
        return emb_action

    def store(self, model_name):
        # torch.save(self.emb.state_dict(), os.path.join('models_action_emb', model_name))
        torch.save({'model_state_dict': self.state_dict()}, os.path.join('models_action_emb', model_name))


class Forward(nn.Module):
    def __init__(self):
        super(Forward, self).__init__()
        self.lin_1 = torch.nn.Linear(232, 200)

    def forward(self, emb_action, enc_state):
        out = torch.cat([emb_action, enc_state], axis=1)
        # creating the one-hot encoded representation
        enc_state = self.lin_1(out)
        return enc_state


class ActionEmbTrainer:
    def __init__(self, encoder, embedding, forward, num_l=18, tensorboard=False):
        self.encoder_nn = encoder
        self.embedding = embedding
        self.forward = forward

        self.num_l = num_l
        self.loss_mse = nn.MSELoss()
        self.loss_fn = lambda pred, targ: torch.sqrt(self.loss_mse(pred, targ))

        params = self.chain(*[self.embedding.parameters(), self.forward.parameters()])
        self.optimizer = torch.optim.RMSprop(params, lr=0.001)

        if tensorboard:
            self.writer = SummaryWriter()
            self.plots = self.plot_tensorboard
        else:
            self.plots = self.no_plots

    def chain(self, *iterables):
        for it in iterables:
            for element in it:
                yield element

    def train_one_epoch(self,
                        epoch_index,
                        training_loader,
                        printing_batch = 10):
        last_loss = 0.
        all_loss = 0.
        self.optimizer.zero_grad(set_to_none=True)

        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            state, action, d, next_state = data
            d.detach()

            enc_state = self.encoder_nn.predict(state)
            embed_a = self.embedding(action)
            next_state_pred = self.forward(embed_a, enc_state)
            next_state_out = self.encoder_nn.predict(next_state)

            # Compute the loss and its gradients
            loss = self.loss_fn(next_state_pred, next_state_out)
            # Gather data and report
            loss.backward()
            # Adjust learning weights
            self.optimizer.step()

            all_loss+= float(loss)

            if i+1 % printing_batch == 0:
                last_loss = all_loss/printing_batch  # loss per batch
                print(f'batch {i+1} loss: {round(last_loss,3)}')
                all_loss=0.

        return last_loss

    def no_plots(self, *args, **kwargs):
        pass

    def plot_tensorboard(self, epoch_index, training_loader, i, last_loss):
        tb_x = epoch_index * len(training_loader) + i + 1
        self.writer.add_scalar('Loss/train', last_loss, tb_x)



