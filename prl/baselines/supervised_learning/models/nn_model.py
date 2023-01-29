import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        current_dim = input_dim

        self.fc1 = torch.nn.Dropout(p=0.2)
        self.fc2 = torch.nn.Linear(input_dim, 512)
        self.fc3 = torch.nn.Dropout(p=0.5)
        self.fc4 = torch.nn.Linear(512,512)
        self.fc5 = torch.nn.Dropout(p=0.5)
        self.fc6 = torch.nn.Linear(512,output_dim)
        self.fc7 = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))
        out = F.relu(self.fc6(out))
        out = self.fc7(out)
        return out

    def log_weights(self):
        """todo: implement as in
        https://github.com/mlflow/mlflow/blob/master/examples/pytorch/mnist_tensorboard_artifact.py"""
