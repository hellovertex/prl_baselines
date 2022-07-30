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
        self.layers = nn.ModuleList()
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = F.softmax(self.layers[-1](x), dim=-1)
        return out

    def log_weights(self):
        """todo: implement as in
        https://github.com/mlflow/mlflow/blob/master/examples/pytorch/mnist_tensorboard_artifact.py"""
