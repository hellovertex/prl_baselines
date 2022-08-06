import logging
import os
import time

import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm

from prl.baselines.supervised_learning.models.nn_model import MLP
from prl.baselines.supervised_learning.training.dataset import OutOfMemoryDataset


def load_checkpoint(path_to_checkpoint):
    return torch.load(path_to_checkpoint)


def run_train_eval(input_dir,
                   epochs,
                   lr,
                   batch_size,
                   test_batch_size,
                   log_interval=100,  # log training metrics every `log_interval` batches
                   ckpt_interval=1000,  # save checkpoint each `ckpt_interval` batches
                   eval_interval=1000,  # eval each `eval_interval` batches
                   ckpt_dir='./ckpt',
                   log_dir='./logdir',
                   resume=True,
                   ):
    # set flags / seeds
    """On Nvidia GPUs you can add the following line at the beginning of our code.
    This will allow the cuda backend to optimize your graph during its first execution.
    However, be aware that if you change the network input/output tensor size the
    graph will be optimized each time a change occurs.
    This can lead to very slow runtime and out of memory errors.
    Only set this flag if your input and output have always the same shape.
    Usually, this results in an improvement of about 20%."""
    torch.backends.cudnn.benchmark = True
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # ckpt_dir to filename for torch save/load functions to work properly
    # datasets tr te
    classes = [0, 1, 2, 3, 4, 5]
    dataset = OutOfMemoryDataset(input_dir, batch_size=batch_size)
    testset = OutOfMemoryDataset(input_dir + '/test', batch_size=batch_size)

    # network
    hidden_dim = [512, 512]
    output_dim = 6
    # waste the first batch to dynamically get the input dimension
    for data in dataset:
        input_dim = data.shape[1] - 1
        break
    # input_dim = 564
    net = MLP(input_dim, output_dim, hidden_dim)

    # if running on GPU and we want to use cuda move model there
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()

    # create optimizers
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    # # load checkpoint if needed/ wanted
    start_n_iter = 0
    start_epoch = 0
    if resume:
        try:
            ckpt = load_checkpoint(ckpt_dir + '/ckpt')
            net.load_state_dict(ckpt['net'])
            start_epoch = ckpt['epoch']
            start_n_iter = ckpt['n_iter']
            optim.load_state_dict(ckpt['optim'])
            print("last checkpoint restored")
        except Exception as e:
            # fail silently and start from scratch
            logging.info(f"Loading checkpoints failed with exception: {e}")
            logging.info(f"Continue Training from scratch")

    # typically we use tensorboardX to keep track of experiments
    writer = SummaryWriter(log_dir=log_dir,
                           comment=f"LR_{lr}_BATCH_{batch_size}")
    n_iter = start_n_iter
    i_train = 0
    total_loss = 0
    correct = 0

    for epoch in range(start_epoch, epochs):
        pbar = tqdm(enumerate(BackgroundGenerator(dataset)), total=len(dataset) / batch_size)
        pbar.set_description(
            f'Training epoch {epoch}/{epochs} on {len(dataset)} examples using batches of size {batch_size}... ')
        start_time = time.time()
        for i, data in pbar:
            labels = data.pop('label')
            data = torch.tensor(data.values)
            labels = torch.tensor(labels.values, dtype=torch.int64)
            if use_cuda:
                data = data.cuda()
                labels = labels.cuda()
            prepare_time = start_time - time.time()

            # forward and backward pass
            optim.zero_grad()
            output = net(data)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            optim.step()
            total_loss += loss.data.item()
            # udpate tensorboardX
            pred = torch.argmax(output, dim=1)  # get the index of the max log-probability
            correct += pred.eq(labels.data).cpu().sum().item()
            i_train += 1
            if i % log_interval == 0:
                writer.add_scalar(tag='Training Loss', scalar_value=total_loss / i_train, global_step=n_iter)
                writer.add_scalar(tag='Training Accuracy', scalar_value=100.0 * correct / i_train, global_step=n_iter)

                i_train = 0
                correct = 0
                total_loss = 0

            # compute computation time and *compute_efficiency*
            process_time = start_time - time.time() - prepare_time
            pbar.set_description("Fraction of NN Training Time: {:.2f}, epoch: {}/{}:".format(
                process_time / (process_time + prepare_time), epoch, epochs))
            start_time = time.time()

            # evaluate
            if i % eval_interval == 0:
                # bring models to evaluation mode
                net.eval()
                # pbar_test = tqdm(enumerate(BackgroundGenerator(dataset)), total=len(testset) / test_batch_size)

                test_loss = 0
                correct = 0
                with torch.no_grad():
                    for j, data in enumerate(BackgroundGenerator(testset)):
                        labels = data.pop('label')
                        data = torch.tensor(data.values)
                        labels = torch.tensor(labels.values, dtype=torch.int64)
                        if use_cuda:
                            data = data.cuda()
                            labels = labels.cuda()
                        output = net(data)
                        test_loss += F.cross_entropy(
                            output, labels, reduction="sum"
                        ).data.item()  # sum up batch loss
                        pred = torch.argmax(output, dim=1)
                        correct += pred.eq(labels.data).cpu().sum().item()

                test_loss /= len(testset)
                test_accuracy = 100.0 * correct / len(testset)
                print(
                    "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                        test_loss, correct, len(testset), test_accuracy
                    )
                )
                # return model to training mode
                net.train()

                # write metrics to tensorboard
                writer.add_scalar(tag='Test Loss', scalar_value=test_loss, global_step=n_iter)
                writer.add_scalar(tag='Test Accuracy', scalar_value=test_accuracy, global_step=n_iter)

                # write layer histograms to tensorboard
                k = 1
                for layer in net.children():
                    if isinstance(layer, nn.Linear):
                        writer.add_histogram(f"layer{k}.weights", layer.state_dict()['weight'], global_step=n_iter)
                        writer.add_histogram(f"layer{k}.bias", layer.state_dict()['bias'], global_step=n_iter)
                        k += 1

                # Write confusion matrix to tensorboard
                prediction = torch.argmax(output, dim=1)
                cf_matrix = confusion_matrix(labels.to('cpu'), prediction.to('cpu'))
                df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in range(output_dim)],
                                     columns=[i for i in classes])
                plt.figure(figsize=(12, 7))
                writer.add_figure("Confusion matrix", sn.heatmap(df_cm, annot=True).get_figure(), epoch)

            # save checkpoint if needed
            if i % ckpt_interval == 0:
                if not os.path.exists(ckpt_dir + '/ckpt'):
                    os.makedirs(ckpt_dir + '/ckpt')

                torch.save({'epoch': epoch,
                            'net': net.state_dict(),
                            'n_iter': n_iter,
                            'optim': optim.state_dict(),
                            'loss': loss}, ckpt_dir + '/ckpt')  # net
                # save model for inference 
                torch.save(net, ckpt_dir + '/model.pt')
