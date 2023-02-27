import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class Encoder(nn.Module):
    def __init__(self, h, w, enc_size=200):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(1, 3, 3), stride=2)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(1, 3, 3), stride=2)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 32, kernel_size=(1, 3, 3), stride=2)
        self.bn3 = nn.BatchNorm3d(32)

        def conv3d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv3d_size_out(conv3d_size_out(conv3d_size_out(w)))
        convh = conv3d_size_out(conv3d_size_out(conv3d_size_out(h)))

        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, enc_size)

    def forward(self, state):
        state = F.relu(self.bn1(self.conv1(state)))
        state = F.relu(self.bn2(self.conv2(state)))
        state = F.relu(self.bn3(self.conv3(state)))
        state = F.relu(self.head(state.view(state.size(0), -1)))
        return state


class Inverse(nn.Module):
    """
    This model tries to capture which action was taken between two frames. In order to
    achieve this is takes as input the embeddings for st and st+1 and returns the probabilities
    for each possible action at.
    """
    def __init__(self, emb_size=200, output_size=18):
        super(Inverse, self).__init__()
        self.lin_1 = nn.Linear(in_features=emb_size*2,
                               out_features=emb_size*4)
        self.head = nn.Linear(in_features=emb_size*4,
                              out_features=output_size)

    def forward(self, emb0, emb1):
        embed = torch.concat([emb0, emb1], axis=1)
        embed = F.relu(self.lin_1(embed))
        logits = F.softmax(self.head(embed), dim=1)
        return logits


class SiamezeTrainer:
    def __init__(self, encoder, inverse, num_l=18, tensorboard=False):
        self.encoder_nn = encoder
        self.inverse_nn = inverse
        self.num_l = num_l
        self.loss_fn = nn.CrossEntropyLoss()

        params = self.chain(*[self.encoder_nn.parameters(), self.inverse_nn.parameters()])
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
                        training_batch = 1):
        last_loss = 0.

        for i, data in enumerate(training_loader):
            # Every data instance is an input + label pair
            state, action, d, next_state = data
            d.detach()
            action = F.one_hot(action, num_classes=18).double()
            emb_0 = self.encoder_nn(state)
            emb_1 = self.encoder_nn(next_state)
            logits = self.inverse_nn(emb_0, emb_1)

            if i+1 % training_batch == 0:
                print(f'train batch {i+1}')
                # Zero your gradients for every batch!
                self.optimizer.zero_grad()
                # Compute the loss and its gradients
                loss = self.loss_fn(logits, action)
                # Gather data and report
                loss.backward()
                # Adjust learning weights
                self.optimizer.step()

                last_loss = float(loss) / training_batch  # loss per batch
                print(f'batch {i+1} loss: {round(last_loss,3)}')
                # self.plots(epoch_index, training_loader, i, last_loss)

            del action, next_state, state, d
        return last_loss

    def no_plots(self, *args, **kwargs):
        pass

    def plot_tensorboard(self, epoch_index, training_loader, i, last_loss):
        tb_x = epoch_index * len(training_loader) + i + 1
        self.writer.add_scalar('Loss/train', last_loss, tb_x)

