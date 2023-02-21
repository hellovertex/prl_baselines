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
from prl.environment.Wrappers.base import ActionSpace
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
import pprint

from prl.baselines.supervised_learning.training.selected_players.dataset import get_datasets
from prl.baselines.supervised_learning.training.utils import get_model, get_model_predict_fold_binary




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
    # return start_n_iter, start_epoch, best_accuracy


def train_eval(abs_input_dir,
               params,
               dichotomize: ActionSpace,
               log_interval,
               eval_interval,
               base_ckptdir,
               base_logdir,
               use_weights=True):
    """abs_intput_dir can be single file or directory that is globbed recursively. in both cases
    and in memory dataset will be created with all csv files found in abs_input_dir and its subfolders.
    """
    target_names = ['Other', dichotomize.name]
    BATCH_SIZE = params['batch_size']
    epochs = params['max_epoch']
    max_env_steps = params['max_env_steps']
    for r in params['rounds']:
        traindataset, testdataset, label_counts = get_datasets(input_dir=abs_input_dir,
                                                               rounds=r,
                                                               dichotomize=dichotomize.value)
        use_cuda = torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
        print('Starting training')

        train_dataloader = DataLoader(traindataset, batch_size=BATCH_SIZE, shuffle=True)
        test_dataloader = DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=True)

        for hdims in params['hdims']:
            for lr in params['lrs']:
                model = get_model_predict_fold_binary(hidden_dims=hdims,
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
                        output = model(x).reshape(-1)
                        pred = output.round()  # get the index of the max log-probability
                        loss = F.binary_cross_entropy(pred, y.type(torch.cuda.FloatTensor))
                        loss.backward()
                        optim.step()
                        total_loss += loss.data.item()  # add batch loss
                        # udpate tensorboardX
                        correct += pred.eq(y.data).cpu().sum().item()
                        i_train += 1
                        # log once (i==0) every epoch (j % log_interval)
                        if j % log_interval == 0 and i == 0:
                            n_batch = i_train * BATCH_SIZE  # how many samples across all batches seen so far
                            writer.add_scalar(tag='Training Loss', scalar_value=total_loss / i_train,
                                              global_step=n_iter)
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

                        # evaluate once (i==0) every epoch (j % eval_interval)
                        if j % eval_interval == 0 and i == 0:
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
                                    output = model(x).reshape(-1)
                                    pred = output.round()  # get the index of the max log-probability
                                    loss = F.binary_cross_entropy(pred, y.type(torch.cuda.FloatTensor))
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
                                            'n_iter': j,  # j * batch_size * len(dataloader) == env_steps
                                            'optim': optim.state_dict(),
                                            'loss': loss,
                                            'best_accuracy': best_accuracy}, ckptdir + '/ckpt.pt')  # net
                                # save model for inference
                                torch.save(model, ckptdir + '/model.pt')
                            else:
                                torch.save({'epoch': epoch,
                                            'net': model.state_dict(),
                                            'n_iter': j,  # j * batch_size * len(dataloader) == env_steps
                                            'optim': optim.state_dict(),
                                            'loss': loss,
                                            'best_accuracy': best_accuracy}, ckptdir + '/ckpt_tmp.pt')  # net
                                # save model for inference
                                torch.save(model, ckptdir + '/model_tmp.pt')
                            # return model to training mode
                            model.train()

                            # write metrics to tensorboard
                            writer.add_scalar(tag='Test Accuracy', scalar_value=test_accuracy, global_step=n_iter)

                            report = classification_report(y.cpu().numpy(),
                                                           pred.cpu().numpy(),
                                                           labels=[0, 1],
                                                           target_names=target_names,
                                                           output_dict=True)

                            for name, values in report.items():
                                if name in target_names:
                                    writer.add_scalar(f'precision/{name}', values['precision'], global_step=n_iter)
                            for name, values in report.items():
                                if name in target_names:
                                    writer.add_scalar(f'recall/{name}', values['recall'], global_step=n_iter)
                            pprint.pprint(report)
                            # write layer histograms to tensorboard
                            k = 1
                            for layer in model.children():
                                if isinstance(layer, nn.Linear):
                                    writer.add_histogram(f"layer{k}.weights", layer.state_dict()['weight'],
                                                         global_step=n_iter)
                                    writer.add_histogram(f"layer{k}.bias", layer.state_dict()['bias'],
                                                         global_step=n_iter)
                                    k += 1
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
               'lrs': [1e-6],  # we ruled out 1e-5 and 1e-7 by hand, 1e-6 is the best we found after multiple trainings
               'rounds': ['preflop', 'flop', 'turn', 'river', 'all'],
               # 'max_epoch': 5_000_000,
               'max_epoch': 100_000_000,
               'max_env_steps': 5_000_000,
               'batch_size': 512}
    # preprocess_flat_data_dir
    abs_path_to_player_data = "/home/hellovertex/Documents/github.com/prl_baselines/data/02_vectorized/top20/with_folds"
    player_dirs = [x[0] for x in os.walk(abs_path_to_player_data)][1:]
    # rounds = 'all'  # use rounds = 'preflop', rounds = 'flop', rounds='turn', rounds='river'
    debug = False
    stem = Path(abs_path_to_player_data).stem
    parent = Path(abs_path_to_player_data).parent.stem
    base_ckptdir = f'./{parent}/only_folds/ckpt_dir'
    base_logdir = f'./{parent}/only_folds/logdir'
    train_eval_fn = partial(train_eval,
                            # abs_input_dir=abs_path,
                            params=params1,
                            dichotomize=ActionSpace.FOLD,
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
