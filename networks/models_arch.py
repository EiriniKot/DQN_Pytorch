from torch import nn
import torch.nn.functional as F


class InverseNet(nn.Module):
    def __init__(self, h, w, outputs = 200):
        super(InverseNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(1, 3, 3), stride=2)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(1, 3, 3), stride=2)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 32, kernel_size=(1, 3, 3), stride=2)
        self.bn3 = nn.BatchNorm3d(32)
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.

        def conv3d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv3d_size_out(conv3d_size_out(conv3d_size_out(w)))
        convh = conv3d_size_out(conv3d_size_out(conv3d_size_out(h)))

        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, y):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        y = F.relu(self.bn1(self.conv1(y)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = F.relu(self.bn3(self.conv3(y)))
        x = self.head(x.view(x.size(0), -1))
        y = x.view(y.size(0), -1)


        return self.head(x)
