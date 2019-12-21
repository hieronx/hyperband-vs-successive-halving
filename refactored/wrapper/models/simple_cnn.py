# https://github.com/pytorch/examples/blob/master/mnist/main.py


import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, tensor_shape, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(tensor_shape[0], 32, 3, 1)
        # nn.init.kaiming_normal_(self.conv1.weight)

        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # nn.init.kaiming_normal_(self.conv2.weight)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(9216, 128)

        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output
