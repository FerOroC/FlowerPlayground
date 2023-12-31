import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        nn.init.ones_(self.conv1.weight)
        nn.init.ones_(self.conv2.weight)
        nn.init.ones_(self.fc1.weight)
        nn.init.ones_(self.fc2.weight)
        nn.init.ones_(self.fc3.weight)
        
        #Change 4 below to min_avail_clients as hyperparam feeding into model Change *
        self.hidden = torch.Tensor()
        self.other_client_params = [torch.zeros(20,6,14,14)] * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        self.hidden = x
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x