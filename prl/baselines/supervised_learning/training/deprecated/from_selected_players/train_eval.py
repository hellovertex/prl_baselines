import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from prefetch_generator import BackgroundGenerator
from prl.environment.Wrappers.augment import ActionSpace
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from prl.baselines.supervised_learning.models.nn_model import MLP
from prl.baselines.supervised_learning.training.dataset import InMemoryDataset
from prl.baselines.supervised_learning.training.utils import init_state, get_in_mem_datasets, get_model


def run_train_eval(input_dir,
                   epochs,
                   lr,
                   batch_size,
                   test_batch_size,
                   log_interval=300,  # log training metrics every `log_interval` train steps
                   ckpt_interval=300,  # save checkpoint each `ckpt_interval` batches
                   eval_interval=300,  # eval each `eval_interval` batches
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
    use_cuda = torch.cuda.is_available()
    traindataset, testdataset = get_in_mem_datasets(input_dir, batch_size)
    train_dataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True)
    traindata, testdata = iter(train_dataloader), iter(test_dataloader)
    hidden_dim = [512, 512]
    model = get_model(traindata, hidden_dims=hidden_dim, device="cpu")
    best_accuracy = test_accuracy = -np.inf
    # create optimizer
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    state_dict = init_state(ckpt_dir, model, optim)
    start_n_iter = state_dict["start_n_iter"]
    start_epoch = state_dict["start_epoch"]
    best_accuracy = state_dict["best_accuracy"]
    # use tensorboardX to keep track of experiments
    writer = SummaryWriter(log_dir=log_dir,
                           comment=f"LR_{lr}_BATCH_{batch_size}")
    n_iter = start_n_iter
    j = 0

    for epoch in range(start_epoch, epochs):
        pbar = tqdm(enumerate(BackgroundGenerator(train_dataloader)), total=round(len(train_dataloader)))
        pbar.set_description(
            f'Training epoch {epoch}/{epochs} on {len(traindataset)} examples using batches of size {batch_size}...')
        start_time = time.time()
        i_train = 0
        correct = 0
        total_loss = 0

        for i, (x, y) in pbar:
            j += 1
            if use_cuda:  # keep
                x = x.cuda()
                y = y.cuda()
            prepare_time = start_time - time.time()

            # forward and backward pass
            optim.zero_grad()
            output = model(x)
            pred = torch.argmax(output, dim=1)  # get the index of the max log-probability
            loss = F.cross_entropy(output, y)
            loss.backward()
            optim.step()
            total_loss += loss.data.item()  # add batch loss
            # udpate tensorboardX
            correct += pred.eq(y.data).cpu().sum().item()
            i_train += 1
            if j % log_interval == 0:
                n_batch = i_train * batch_size  # how many samples across all batches seen so far
                writer.add_scalar(tag='Training Loss', scalar_value=total_loss / i_train, global_step=n_iter)
                writer.add_scalar(tag='Training Accuracy', scalar_value=100.0 * correct / n_batch, global_step=n_iter)
                print(f"\nTrain set: "
                      f"Average loss: {round(total_loss / i_train, 4)}, "
                      f"Accuracy: {correct}/{n_batch} ({round(100.0 * correct / n_batch, 2)}%)\n")
                i_train = 0
                correct = 0
                total_loss = 0

            # compute computation time and *compute_efficiency*
            process_time = start_time - time.time() - prepare_time
            try:
                fraction = process_time / (process_time + prepare_time)
            except ZeroDivisionError:
                fraction = 0
            # note that we prioritize GPU usage and iterations per second over Fraction of NN traintime
            pbar.set_description("Fraction of NN Training Time: {:.2f}, epoch: {}/{}:".format(
                fraction, epoch, epochs))
            start_time = time.time()

            # evaluate
            if j % eval_interval == 0:
                # bring models to evaluation mode
                model.eval()
                # pbar_test = tqdm(enumerate(BackgroundGenerator(dataset)), total=len(testset) / test_batch_size)

                test_loss = 0
                test_correct = 0
                with torch.no_grad():
                    for x, y in BackgroundGenerator(test_dataloader):
                        if use_cuda:
                            x = x.cuda()
                            y = y.cuda()
                        output = model(x)
                        # sum up batch loss
                        test_loss += F.cross_entropy(output, y, reduction="sum").data.item()
                        pred = torch.argmax(output, dim=1)
                        test_correct += pred.eq(y.data).cpu().sum().item()

                test_loss /= len(testdataset)
                test_accuracy = 100 * test_correct / len(testdataset)
                print(
                    "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                        test_loss, round(test_correct), len(testdataset), test_accuracy
                    )
                )
                if not os.path.exists(ckpt_dir + '/ckpt'):
                    os.makedirs(ckpt_dir + '/ckpt')
                if best_accuracy < test_accuracy:
                    best_accuracy = test_accuracy
                    torch.save({'epoch': epoch,
                                'net': model.state_dict(),
                                'n_iter': n_iter,
                                'optim': optim.state_dict(),
                                'loss': loss,
                                'best_accuracy': best_accuracy}, ckpt_dir + '/ckpt.pt')  # net
                    # save model for inference
                    torch.save(model, ckpt_dir + '/model.pt')
                # return model to training mode
                model.train()

                # write metrics to tensorboard
                writer.add_scalar(tag='Test Loss', scalar_value=test_loss, global_step=n_iter)
                writer.add_scalar(tag='Test Accuracy', scalar_value=test_accuracy, global_step=n_iter)

                # write layer histograms to tensorboard
                k = 1
                for layer in model.children():
                    if isinstance(layer, nn.Linear):
                        writer.add_histogram(f"layer{k}.weights", layer.state_dict()['weight'], global_step=n_iter)
                        writer.add_histogram(f"layer{k}.bias", layer.state_dict()['bias'], global_step=n_iter)
                        k += 1
            n_iter += 1
