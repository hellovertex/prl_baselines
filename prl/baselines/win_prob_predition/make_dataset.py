import multiprocessing
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
from prl.environment.Wrappers.base import ActionSpace
from tqdm import tqdm

from prl.baselines.cpp_hand_evaluator.monte_carlo import HandEvaluator_MonteCarlo

IDX_C0_0 = 167  # feature_names.index('0th_player_card_0_rank_0')
IDX_C0_1 = 184  # feature_names.index('0th_player_card_1_rank_0')
IDX_C1_0 = 184  # feature_names.index('0th_player_card_1_rank_0')
IDX_C1_1 = 201  # feature_names.index('1th_player_card_0_rank_0')
IDX_BOARD_START = 82  #
IDX_BOARD_END = 167  #
CARD_BITS_TO_STR = np.array(
    ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c'])
BOARD_BITS_TO_STR = np.array(
    ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
     'h', 'd', 's', 'c', '2', '3', '4', '5', '6', '7', '8', '9', 'T',
     'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c', '2', '3', '4', '5', '6',
     '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c', '2',
     '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h',
     'd', 's', 'c', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J',
     'Q', 'K', 'A', 'h', 'd', 's', 'c'])
RANK = 0
SUITE = 1

from prl.baselines.examples.examples_tianshou_env import make_default_tianshou_env


def main(out_file_suffix: int, num_players, max_episodes_per_file):
    env = make_default_tianshou_env(mc_model_ckpt_path=None,  # dont use mc
                                    stack_sizes=[20000 for _ in
                                                 range(num_players)],
                                    agents=[f'p{i}' for i in range(num_players)],
                                    num_players=num_players)
    observations = np.zeros((max_episodes_per_file, 569))
    labels = np.zeros((max_episodes_per_file, 1))
    mc_eval = HandEvaluator_MonteCarlo()
    n_iter = 5000
    # from cards get winning probabilityx
    for i in tqdm(range(max_episodes_per_file)):
        obs_dict = env.reset()
        while True:
            if obs_dict['mask'][8] == 1:
                action = ActionSpace.NoOp
            else:
                action = ActionSpace.CHECK_CALL
            obs_dict, rews, terminated, truncated, info = env.step(action)
            obs = obs_dict['obs']
            mc_dict = mc_eval.run_mc(obs,
                                     n_opponents=num_players - 1,
                                     n_iter=n_iter)
            win_prob = (mc_dict['won'] + mc_dict['tied']) / n_iter
            observations[i, :] = obs
            labels[i] = win_prob
            if terminated:
                break
    # save to disk
    outpath = './dataset'
    if not os.path.exists(Path(outpath)):
        os.makedirs(outpath, exist_ok=True)
    np.save(os.path.join(*[
        outpath,
        f'data_{out_file_suffix}.npy'
    ]), observations)
    np.save(os.path.join(*[
        outpath,
        f'labels_{out_file_suffix}.npy'
    ]), labels)


if __name__ == '__main__':
    max_files = 20
    # 1a) run games using 2 calling stations
    start = time.time()
    p = multiprocessing.Pool()
    t0 = time.time()
    fn = partial(main,
                 num_players=2, max_episodes_per_file=20000)
    for x in p.imap_unordered(fn, [i for i in range(max_files)]):
        print(x + f'. Took {time.time() - t0} seconds')
    print(f'Finished job after {time.time() - start} seconds.')
    p.close()

    # # 1b). run games using 6 calling stations
    # start = time.time()
    # p = multiprocessing.Pool()
    # t0 = time.time()
    # fn = partial(main,
    #              num_players=6, max_episodes_per_file=20000)
    # for x in p.imap_unordered(fn, [6000+i for i in range(max_files)]):
    #     print(x + f'. Took {time.time() - t0} seconds')
    # print(f'Finished job after {time.time() - start} seconds.')
    # p.close()
