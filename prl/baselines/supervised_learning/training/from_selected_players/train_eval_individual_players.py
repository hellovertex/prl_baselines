import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from prefetch_generator import BackgroundGenerator
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from prl.baselines.supervised_learning.training.utils import init_state, get_in_mem_datasets, get_model


def train_eval(abs_input_dir, params, log_interval, eval_interval):
    leaf_dir = abs_input_dir.split('/')[-1]
    base_logdir = f'./logdir/{leaf_dir}'
    base_ckptdir = f'./ckpt_dir/{leaf_dir}'
    BATCH_SIZE = params['batch_size']
    traindataset, testdataset = get_in_mem_datasets(abs_input_dir, BATCH_SIZE)
    train_dataloader = DataLoader(traindataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=True)
    traindata, testdata = iter(train_dataloader), iter(test_dataloader)
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    epochs = params['max_epoch']
    for hdims in params['hdims']:
        model = get_model(traindata, hidden_dims=hdims, device=device)
        for lr in params['lrs']:
            logdir = base_logdir + f'_{hdims}_{lr}'
            ckptdir = base_ckptdir + f'_{hdims}_{lr}'
            optim = torch.optim.Adam(model.parameters(), lr=lr)
            state_dict = init_state(ckptdir, model, optim)
            start_n_iter = state_dict["start_n_iter"]
            start_epoch = state_dict["start_epoch"]
            best_accuracy = state_dict["best_accuracy"]
            writer = SummaryWriter(log_dir=logdir)
            n_iter = start_n_iter
            j = 0
            for epoch in range(start_epoch, epochs):
                pbar = tqdm(enumerate(BackgroundGenerator(train_dataloader)),
                            total=round(len(train_dataloader)))
                pbar.set_description(
                    f'Training epoch {epoch}/{epochs} on {len(traindataset)} '
                    f'examples using batches of size {BATCH_SIZE}...')
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
                        n_batch = i_train * BATCH_SIZE  # how many samples across all batches seen so far
                        writer.add_scalar(tag='Training Loss', scalar_value=total_loss / i_train, global_step=n_iter)
                        writer.add_scalar(tag='Training Accuracy', scalar_value=100.0 * correct / n_batch,
                                          global_step=n_iter)
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
                        model.eval()
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
                        if not os.path.exists(ckptdir + '/ckpt'):
                            os.makedirs(ckptdir + '/ckpt')
                        if best_accuracy < test_accuracy:
                            best_accuracy = test_accuracy
                            torch.save({'epoch': epoch,
                                        'net': model.state_dict(),
                                        'n_iter': n_iter,
                                        'optim': optim.state_dict(),
                                        'loss': loss,
                                        'best_accuracy': best_accuracy}, ckptdir + '/ckpt.pt')  # net
                            # save model for inference
                            torch.save(model, ckptdir + '/model.pt')
                        # return model to training mode
                        model.train()

                        # write metrics to tensorboard
                        writer.add_scalar(tag='Test Loss', scalar_value=test_loss, global_step=n_iter)
                        writer.add_scalar(tag='Test Accuracy', scalar_value=test_accuracy, global_step=n_iter)

                        # write layer histograms to tensorboard
                        k = 1
                        for layer in model.children():
                            if isinstance(layer, nn.Linear):
                                writer.add_histogram(f"layer{k}.weights", layer.state_dict()['weight'],
                                                     global_step=n_iter)
                                writer.add_histogram(f"layer{k}.bias", layer.state_dict()['bias'], global_step=n_iter)
                                k += 1
                    n_iter += 1


if __name__ == "__main__":
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

    # export TRAIN_EVAL_SOURCE_DIR=/home/.../Documents/github.com/prl_baselines/data/02_vectorized/0.25-0.50/...
    # filenames = glob.glob(os.environ["TRAIN_EVAL_SOURCE_DIR"]+"/**/*.txt",recursive=True)
    log_interval = eval_interval = 100  # trainsteps (a BATCH_SIZE observations)
    params = {'hdims': [[256], [512], [256, 256], [512, 512]],
              'lrs': [1e-6, 1e-5, 1e-7],
              'max_epoch': 5_000_000,
              'batch_size': 512,
              }
    player_dirs = [x[0] for x in os.walk(os.environ["TRAIN_EVAL_SOURCE_DIR"])]
    for player_subdir in player_dirs:
        train_eval(player_subdir, params, log_interval, eval_interval)
