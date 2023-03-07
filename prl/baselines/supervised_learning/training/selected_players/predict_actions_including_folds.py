# parameterize the round for which actions should be predicted
# possible values are [all_rounds, preflop, flop, turn, river]
# hard code the data that is being loaded at start from top20/with_folds


import logging
import multiprocessing
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from prefetch_generator import BackgroundGenerator
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
import pprint

from prl.baselines.supervised_learning.training.selected_players.dataset import \
    get_datasets
from prl.baselines.supervised_learning.training.utils import get_model

target_names = ['Fold',
                'Check Call',
                'Raise Third Pot',
                'Raise Two Thirds Pot',
                'Raise Pot',
                'Raise 2x Pot',
                'Raise 3x Pot',
                'Raise All in']


def init_state(ckpt_dir, model, optim, resume: bool = True):
    # # load checkpoint if needed/ wanted

    start_n_iter = 0
    start_epoch = 0
    best_accuracy = -np.inf
    if resume:
        try:
            ckpt = torch.load(ckpt_dir + '/ckpt.pt')
            model.load_state_dict(ckpt['net'])
            start_epoch = ckpt['epoch']
            start_n_iter = ckpt['n_iter']
            best_accuracy = ckpt['best_accuracy']
            optim.load_state_dict(ckpt['optim'])
            print("last checkpoint restored")
        except Exception as e:
            # fail silently and start from scratch
            logging.info(f"Loading checkpoints failed with exception: {e}")
            logging.info(f"Continue Training from scratch")
    return {"start_n_iter": start_n_iter,
            "start_epoch": start_epoch,
            "best_accuracy": best_accuracy}


def train_eval(abs_input_dir,
               params,
               log_interval,
               eval_interval,
               base_ckptdir,
               base_logdir,
               use_weights=True):
    """abs_intput_dir can be single file or directory that is globbed recursively. in both cases
    and in memory dataset will be created with all csv files found in abs_input_dir and its subfolders.
    """
    BATCH_SIZE = params['batch_size']
    epochs = params['max_epoch']
    max_env_steps = params['max_env_steps']
    for r in params['rounds']:
        traindataset, testdataset, label_counts = get_datasets(input_dir=abs_input_dir,
                                                               rounds=r,
                                                               )
        use_cuda = torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
        print('Starting training')
        weights = None
        if use_weights:
            weights.to(device)

        train_dataloader = DataLoader(traindataset, batch_size=BATCH_SIZE, shuffle=True)
        test_dataloader = DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=True)
        traindata, testdata = iter(train_dataloader), iter(test_dataloader)

        for hdims in params['hdims']:
            for lr in params['lrs']:
                model = get_model(traindata, output_dim=8, hidden_dims=hdims,
                                  device=device)
                logdir = base_logdir + f'/{r}/{Path(abs_input_dir).stem}/{hdims}_{lr}'
                ckptdir = base_ckptdir + f'/{r}/{Path(abs_input_dir).stem}/{hdims}_{lr}'
                optim = torch.optim.Adam(model.parameters(),
                                         lr=lr)
                state_dict = init_state(ckptdir, model, optim)
                start_n_iter = state_dict["start_n_iter"]
                start_epoch = state_dict["start_epoch"]
                best_accuracy = state_dict["best_accuracy"]
                writer = SummaryWriter(log_dir=logdir)
                n_iter = start_n_iter
                j = start_n_iter
                # todo: refactor into run_train_loop(model, optim, writer, train_options)
                for epoch in range(start_epoch, epochs):
                    len_data = round(len(train_dataloader))
                    env_steps = j * len_data
                    if env_steps > max_env_steps:
                        break
                    pbar = tqdm(enumerate(BackgroundGenerator(train_dataloader)),
                                total=len_data)
                    pbar.set_description(
                        f'Training epoch {epoch}/{epochs} on {len(traindataset)} '
                        f'examples using batches of size {BATCH_SIZE}...')
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
                        pred = torch.argmax(output,
                                            dim=1)  # get the index of the max log-probability
                        if weights is not None:
                            loss = F.cross_entropy(output, y, weight=weights.to(device))
                        else:
                            loss = F.cross_entropy(output, y)
                        loss.backward()
                        optim.step()
                        total_loss += loss.data.item()  # add batch loss
                        # udpate tensorboardX
                        correct += pred.eq(y.data).cpu().sum().item()
                        i_train += 1
                        f1 = f1_score(y.data.cpu(), pred.cpu(), average='weighted')
                        # log once (i==0) every epoch (j % log_interval)
                        if j % log_interval == 0 and i == 0:
                            n_batch = i_train * BATCH_SIZE  # how many samples across all batches seen so far
                            writer.add_scalar(tag='Training Loss',
                                              scalar_value=total_loss / i_train,
                                              global_step=n_iter)
                            writer.add_scalar(tag='Training F1 score', scalar_value=f1,
                                              global_step=n_iter)
                            writer.add_scalar(tag='Training Accuracy',
                                              scalar_value=100.0 * correct / n_batch,
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
                        pbar.set_description(
                            "Fraction of NN Training Time: {:.2f}, epoch: {}/{}:".format(
                                fraction, epoch, epochs))
                        start_time = time.time()


                        n_iter += 1
                    j += 1

    return f"Finished training from {abs_input_dir}. " \
           f"Logs and checkpoints can be found under {base_logdir} and {base_ckptdir} resp."


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


    # disable sklearn warnings
    def warn(*args, **kwargs):
        pass


    import warnings

    warnings.warn = warn

    log_interval = eval_interval = 5
    params1 = {'hdims': [[512]],  # [256, 256], [512, 512]], -- not better
               'lrs': [1e-6],
               # we ruled out 1e-5 and 1e-7 by hand, 1e-6 is the best we found after multiple trainings
               'rounds': ['flop', 'turn', 'river', 'all', 'preflop'],
               # 'max_epoch': 5_000_000,
               'max_epoch': 100_000_000,
               'max_env_steps': 1_000_000,
               'batch_size': 512}
    # preprocess_flat_data_dir
    abs_path_to_player_data = "/home/hellovertex/Documents/github.com/prl_baselines/data/02_vectorized/top20/with_folds"
    player_dirs = [x[0] for x in os.walk(abs_path_to_player_data)][1:]
    # rounds = 'all'  # use rounds = 'preflop', rounds = 'flop', rounds='turn', rounds='river'
    debug = False
    stem = Path(abs_path_to_player_data).stem
    parent = Path(abs_path_to_player_data).parent.stem
    base_ckptdir = f'./{parent}/including_folds/{stem}/ckpt_dir'
    base_logdir = f'./{parent}/including_folds/{stem}/logdir'
    train_eval_fn = partial(train_eval,
                            # abs_input_dir=abs_path,
                            params=params1,
                            # rounds=rounds,
                            log_interval=log_interval,
                            eval_interval=eval_interval,
                            base_ckptdir=base_ckptdir,
                            base_logdir=base_logdir)
    if debug:
        for player_subdir in player_dirs:
            train_eval_fn(abs_input_dir=player_subdir)
    else:
        start = time.time()
        p = multiprocessing.Pool()
        t0 = time.time()
        # train x NNs at once
        x = 3
        chunks = []
        current_chunk = []
        i = 0
        for subdir in player_dirs:
            current_chunk.append(subdir)
            if (i + 1) % x == 0:
                chunks.append(current_chunk)
                current_chunk = []
            i += 1
        for dirs in chunks:
            for x in p.imap_unordered(train_eval_fn, dirs):
                print(x + f'. Took {time.time() - t0} seconds')
            print(f'Finished job after {time.time() - start} seconds.')
        p.close()
