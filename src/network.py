import torch
from torch import nn
import torch.nn.functional as F
from src.siameze import Encoder
from src.action_embeddings import EmbeddingModel
from src.nn_utils import ModelLoader


class DqnNet(nn.Module):
    def __init__(self, h, w, outputs):
        super(DqnNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(1, 3, 3), stride=2)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=2)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 32, kernel_size=(1, 3, 3), stride=2)
        self.bn3 = nn.BatchNorm3d(32)
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.

        def conv3d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv3d_size_out(conv3d_size_out(conv3d_size_out(w)))
        convh = conv3d_size_out(conv3d_size_out(conv3d_size_out(h)))

        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.softmax(self.head(x), dim=1)
        return x


class DqnNetAlternative(nn.Module):
    def __init__(self,
                 h, w,
                 enc_size,
                 emb_depth,
                 n_actions):

        super(DqnNetAlternative, self).__init__()
        encoder = Encoder(h=h, w=w, enc_size=enc_size)
        embedding = EmbeddingModel(num_embeddings=n_actions, embedding_dim=emb_depth)

        self.enc_load = ModelLoader(path='models_inverse_encoded/checkpoint_0_encoder.pt',
                                    model_to_load=encoder,
                                    frozen=False)

        self.embedding_nn = ModelLoader(path='models_action_emb/actions_embedding.pt',
                                        model_to_load=embedding,
                                        frozen=False)

        self.lin_0 = nn.Linear(in_features=enc_size, out_features=emb_depth)
        self.lin_1 = nn.Linear(in_features=emb_depth*n_actions, out_features=emb_depth)
        self.multihead = nn.MultiheadAttention(embed_dim=enc_size,
                                               num_heads=4,
                                               kdim=emb_depth,
                                               vdim=emb_depth,
                                               dropout=0.1,
                                               batch_first=True)
        self.head = nn.Linear(enc_size, n_actions)

    def forward(self, x):
        x = self.enc_load.predict(x)
        x = torch.unsqueeze(x, 1)
        emb = self.embedding_nn.model_loaded.emb.weight
        emb = torch.unsqueeze(emb, 0).repeat(x.shape[0],1,1)
        x,  _ = self.multihead(x, emb, emb, need_weights=False)
        x = F.softmax(torch.squeeze(self.head(x), dim=1),
                      dim=1)
        return x

