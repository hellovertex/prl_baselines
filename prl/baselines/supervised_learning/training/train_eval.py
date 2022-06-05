import logging
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from prefetch_generator import BackgroundGenerator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from prl.baselines.supervised_learning.models.model import Net
from prl.baselines.supervised_learning.training.dataset import OutOfMemoryDataset

BATCH_SIZE = 1000


def load_checkpoint(path_to_checkpoint):
    return torch.load(path_to_checkpoint)


def run_train_eval(input_dir,
                   epochs,
                   lr,
                   ckpt_interval=10000,  # save checkpoint each `ckpt_interval` batches
                   eval_interval=10000,  # eval each `eval_interval` batches
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

    # datasets tr te
    dataset = OutOfMemoryDataset(input_dir, batch_size=BATCH_SIZE)
    testset = OutOfMemoryDataset(input_dir+'/test', batch_size=BATCH_SIZE)

    # network
    hidden_dim = [512, 512]
    net = Net(564, 6, hidden_dim)

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
            ckpt = load_checkpoint(ckpt_dir)
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
    writer = SummaryWriter(log_dir=log_dir)
    n_iter = start_n_iter
    pbar = tqdm(enumerate(BackgroundGenerator(dataset)), total=len(dataset) / BATCH_SIZE)
    for epoch in range(start_epoch, epochs):
        pbar.set_description(
            f'Training epoch {epoch}/{epochs} on {len(dataset)} examples using batches of size {BATCH_SIZE}... ')
        start_time = time.time()
        for i, data in pbar:
            # todo convert to pytorch tensors if applicable:
            label = data.pop['label']
            if use_cuda:
                data = data.cuda()
                label = label.cuda()
            prepare_time = start_time - time.time()

            # forward and backward pass
            optim.zero_grad()
            output = net(data)
            # todo check if this works with our batches
            loss = F.cross_entropy(output, label)
            loss.backward()
            loss.backward()
            optim.step()

            # udpate tensorboardX
            # todo: scalar_value
            writer.add_scalar(tag='', scalar_value=0, global_step=n_iter)

            # compute computation time and *compute_efficiency*
            process_time = start_time - time.time() - prepare_time
            pbar.set_description("Compute efficiency: {:.2f}, epoch: {}/{}:".format(
                process_time / (process_time + prepare_time), epoch, epochs))
            start_time = time.time()

            # evaluate
            if eval_interval % i == 0:
                # bring models to evaluation mode
                net.eval()
                pbar_test = tqdm(enumerate(BackgroundGenerator(dataset)), total=len(testset) / BATCH_SIZE)

                test_loss = 0
                correct = 0
                with torch.no_grad():
                    for j, data in pbar_test:
                        label = data.pop['label']
                        if use_cuda:
                            data = data.cuda()
                            label = label.cuda()
                        output = net(data)
                        test_loss += F.cross_entropy(
                            output, label, reduction="sum"
                        ).data.item()  # sum up batch loss
                        pred = output.data.max(1)[1]  # get the index of the max log-probability
                        correct += pred.eq(label.data).cpu().sum().item()

                test_loss /= len(testset)
                test_accuracy = 100.0 * correct / len(testset)
                print(
                    "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                        test_loss, correct, len(testset), test_accuracy
                    )
                )
                # return model to training mode
                net.train()
                # todo: write test metrics to tensorboard summarywriter
                writer.add_scalar(tag='', scalar_value=0, global_step=n_iter)
            # save checkpoint if needed
            if ckpt_interval % i == 0:
                torch.save({'epoch': epoch,
                            'net': net.state_dict(),
                            'n_iter': n_iter,
                            'optim': optim.state_dict(),
                            'loss': loss}, ckpt_dir)  # net


