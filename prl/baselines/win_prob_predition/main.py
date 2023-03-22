import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tianshou.utils.net.common import MLP
from torch.utils.data import TensorDataset


def load_model(ckpt_path=None, flatten_input=False, device='cpu'):
    input_dim = 569
    output_dim = 1
    hidden_dims = [256]
    model = MLP(input_dim,
                output_dim,
                hidden_dims,
                flatten_input=flatten_input).to(device)
    if ckpt_path:
        ckpt = torch.load(ckpt_path,
                          map_location=device)
        model.load_state_dict(ckpt['net'])
    model.eval()
    return model


def load_data(data_dir, n_files=20, start_idx=0):
    data = []
    labels = []
    files = glob.glob(data_dir + '**/*.npy*', recursive=False)
    for i in range(start_idx, n_files + 1):
        d = f'data_{i}.npy'
        l = f'labels_{i}.npy'
        len_data = len(data)
        len_labels = len(labels)
        for f in files:
            if d in f:
                data.append(np.load(f))
            if l in f:
                labels.append(np.load(f))
        assert len(data) == len_data + 1
        assert len(labels) == len_labels + 1
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    return data, labels


def main(data, labels):
    learning_rate = 1e-6
    n_epochs = 100
    criterion = nn.MSELoss()
    net = load_model()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    batch_size = 512
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels,
                                                                        test_size=0.05)
    train_data = torch.from_numpy(train_data).float()
    train_labels = torch.from_numpy(train_labels).float()
    test_data = torch.from_numpy(test_data).float()
    test_labels = torch.from_numpy(test_labels).float()
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True)
    with torch.no_grad():
        pred = net(test_data)
        loss = criterion(pred, test_labels)
        variance = torch.var(test_labels)
        r2_score = 1 - loss / variance
        print(f'Epoch {0}: MSE loss = {loss:.4f}')
        print(f'Epoch {0}: R2 score = {r2_score:.4f}')
        # Create a scatter plot
        plt.scatter(pred, test_labels)
        plt.xlabel('True probabilities')
        plt.ylabel('Predicted probabilities')
        plt.title('Scatter plot of true probabilities vs predicted probabilities')
        x = np.linspace(0, 1, 100)
        plt.plot(x, x, color='red')
        plt.show()

    for epoch in range(1, n_epochs):
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
        outputs = net(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        print('Epoch {}, Loss: {:.4f}'.format(epoch + 1, loss.item()))
        # Compute accuracy on test set
        with torch.no_grad():
            pred = net(test_data)
            loss = criterion(pred, test_labels)
            r2_score = 1 - loss / variance
            print(f'Epoch {epoch}: MSE loss = {loss:.4f}')
            print(f'Epoch {epoch}: R2 score = {r2_score:.4f}')
            if epoch % 5 == 0:
                plt.scatter(pred, test_labels)
                plt.xlabel('True probabilities')
                plt.ylabel('Predicted probabilities')
                plt.title('Scatter plot of true probabilities vs predicted probabilities')
                x = np.linspace(0, 1, 100)
                plt.plot(x, x, color='red')
                plt.show()


if __name__ == '__main__':
    data_dir = './data'
    data, labels = load_data(data_dir, n_files=2, start_idx=1)
    main(data, labels)
