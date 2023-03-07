import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_old(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(MLP_old, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = self.layers[-1](x)
        return out

    def log_weights(self):
        """todo: implement as in
        https://github.com/mlflow/mlflow/blob/master/examples/pytorch/mnist_tensorboard_artifact.py"""


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim):

        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        # quick and dirty but yet good
        layer_size = hidden_dim[0]
        self.fc1 = torch.nn.Dropout(p=0.2)
        self.fc2 = torch.nn.Linear(input_dim, layer_size)
        if len(hidden_dim) == 2:
            self.fc3 = torch.nn.Dropout(p=0.5)
            self.fc4 = torch.nn.Linear(layer_size, layer_size)
        self.fc5 = torch.nn.Dropout(p=0.5)
        self.fc6 = torch.nn.Linear(layer_size, output_dim)
        self.fc7 = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        if len(self.hidden_dim) == 2:
            out = F.relu(self.fc3(out))
            out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))
        out = F.relu(self.fc6(out))
        out = self.fc7(out)
        return out

    def log_weights(self):
        """todo: implement as in
        https://github.com/mlflow/mlflow/blob/master/examples/pytorch/mnist_tensorboard_artifact.py"""
