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


def load_checkpoint(path_to_checkpoint):
    return torch.load(path_to_checkpoint)


def get_datasets(input_dir, batch_size):
    dataset = InMemoryDataset(input_dir)
    total_len = len(dataset)
    train_len = int(total_len * 0.8)
    test_len = int(total_len * 0.1)
    val_len = int(total_len * 0.1)
    # add residuals to val_len to add up to total_len
    val_len += total_len - (int(train_len) + int(test_len) + int(val_len))
    # set seed
    gen = torch.Generator().manual_seed(1)
    train, test, val = random_split(dataset, [train_len, test_len, val_len], generator=gen)

    return train, test


def get_model(traindata):
    # network
    classes = [ActionSpace.FOLD,
               ActionSpace.CHECK_CALL,  # CHECK IS INCLUDED
               ActionSpace.RAISE_MIN_OR_3BB,
               ActionSpace.RAISE_HALF_POT,
               ActionSpace.RAISE_POT,
               ActionSpace.ALL_IN]
    hidden_dim = [512, 512]
    output_dim = len(classes)
    input_dim = None
    # waste the first batch to dynamically get the input dimension
    for x, y in traindata:
        input_dim = x.shape[1]
        break
    net = MLP(input_dim, output_dim, hidden_dim)
    # if running on GPU and we want to use cuda move model there
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
    return net


def init_state(ckpt_dir, resume: bool, model, optim):
    # # load checkpoint if needed/ wanted

    start_n_iter = 0
    start_epoch = 0
    if resume:
        try:
            ckpt = load_checkpoint(ckpt_dir + '/ckpt.pt')
            model.load_state_dict(ckpt['net'])
            start_epoch = ckpt['epoch']
            start_n_iter = ckpt['n_iter']
            optim.load_state_dict(ckpt['optim'])
            print("last checkpoint restored")
        except Exception as e:
            # fail silently and start from scratch
            logging.info(f"Loading checkpoints failed with exception: {e}")
            logging.info(f"Continue Training from scratch")
    return start_n_iter, start_epoch


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
    use_cuda = torch.cuda.is_available()
    # ckpt_dir to filename for torch save/load functions to work properly
    # datasets tr te

    traindataset, testdataset = get_datasets(input_dir, batch_size)
    train_dataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True)
    traindata, testdata = iter(train_dataloader), iter(test_dataloader)
    model = get_model(traindata)

    # create optimizer
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    start_n_iter, start_epoch = init_state(ckpt_dir, resume, model, optim)

    # use tensorboardX to keep track of experiments
    writer = SummaryWriter(log_dir=log_dir,
                           comment=f"LR_{lr}_BATCH_{batch_size}")
    n_iter = start_n_iter
    i_train = 0
    total_loss = 0
    correct = 0

    for epoch in range(start_epoch, epochs):
        pbar = tqdm(enumerate(BackgroundGenerator(train_dataloader)), total=round(len(train_dataloader)))
        pbar.set_description(
            f'Training epoch {epoch}/{epochs} on {len(traindataset)} examples using batches of size {batch_size}...')
        start_time = time.time()
        i_train = 0
        correct = 0
        total_loss = 0

        for i, (x, y) in pbar:
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
            if i % log_interval == 0:
                n_batch = i_train * batch_size  # how many samples across all batches seen so far
                writer.add_scalar(tag='Training Loss', scalar_value=total_loss / i_train, global_step=n_iter)
                writer.add_scalar(tag='Training Accuracy', scalar_value=100.0 * correct / n_batch, global_step=n_iter)
                print(f"\nTrain set: "
                      f"Average loss: {round(total_loss / i_train, 4)}, "
                      f"Accuracy: {correct}/{n_batch} ({round(100.0 * correct / n_batch, 2)}%)\n")
                if correct > 100000:
                    print('wtf')
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
            if i % eval_interval == 0:
                # bring models to evaluation mode
                model.eval()
                # pbar_test = tqdm(enumerate(BackgroundGenerator(dataset)), total=len(testset) / test_batch_size)

                test_loss = 0
                test_correct = 0
                with torch.no_grad():
                    for j, (x, y) in enumerate(BackgroundGenerator(test_dataloader)):
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

            # save checkpoint if needed
            if i % ckpt_interval == 0:
                if not os.path.exists(ckpt_dir + '/ckpt'):
                    os.makedirs(ckpt_dir + '/ckpt')

                torch.save({'epoch': epoch,
                            'net': model.state_dict(),
                            'n_iter': n_iter,
                            'optim': optim.state_dict(),
                            'loss': loss}, ckpt_dir + '/ckpt.pt')  # net
                # save model for inference 
                torch.save(model, ckpt_dir + '/model.pt')
            n_iter += 1
