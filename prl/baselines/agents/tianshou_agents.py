import os
from random import random
from typing import Tuple, List, Union

import numpy as np
import torch
from prl.environment.Wrappers.base import ActionSpace

from prl.baselines.cpp_hand_evaluator.monte_carlo import HandEvaluator_MonteCarlo
from prl.baselines.cpp_hand_evaluator.rank import dict_str_to_sk
from prl.baselines.supervised_learning.models.nn_model import MLP, MLP_old

IDX_C0_0 = 167  # feature_names.index('0th_player_card_0_rank_0')
IDX_C0_1 = 184  # feature_names.index('0th_player_card_1_rank_0')
IDX_C1_0 = 184  # feature_names.index('0th_player_card_1_rank_0')
IDX_C1_1 = 201  # feature_names.index('1th_player_card_0_rank_0')
IDX_BOARD_START = 82  #
IDX_BOARD_END = 167  #
CARD_BITS_TO_STR = np.array(['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c'])
BOARD_BITS_TO_STR = np.array(['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
                              'h', 'd', 's', 'c', '2', '3', '4', '5', '6', '7', '8', '9', 'T',
                              'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c', '2', '3', '4', '5', '6',
                              '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c', '2',
                              '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h',
                              'd', 's', 'c', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J',
                              'Q', 'K', 'A', 'h', 'd', 's', 'c'])
RANK = 0
SUITE = 1


class MCAgent:

    def __init__(self):
        self._mc_iters = 5000
        self.tightness = .9
        self._card_evaluator = HandEvaluator_MonteCarlo()
        self.load_model()

    def load_model(self):
        input_dim = 564
        classes = [ActionSpace.FOLD,
                   ActionSpace.CHECK_CALL,  # CHECK IS INCLUDED in CHECK_CALL
                   ActionSpace.RAISE_MIN_OR_3BB,
                   ActionSpace.RAISE_HALF_POT,
                   ActionSpace.RAISE_POT,
                   ActionSpace.ALL_IN]
        hidden_dim = [512, 512]
        output_dim = len(classes)
        net = MLP(input_dim, output_dim, hidden_dim)
        # if running on GPU and we want to use cuda move model there
        # use_cuda = torch.cuda.is_available()
        # if use_cuda:
        #     net = net.cuda()
        self._model = net
        os.environ[
            'PRL_BASELINE_MODEL_PATH'] = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/ckpt/ckpt.pt"
        ckpt = torch.load(os.environ['PRL_BASELINE_MODEL_PATH'],
                          map_location=torch.device('cpu'))
        self._model.load_state_dict(ckpt['net'])
        self._model.eval()
        return net

    def card_bit_mask_to_int(self, c0: np.array, c1: np.array, board_mask: np.array) -> Tuple[List[int], List[int]]:
        c0_1d = dict_str_to_sk[CARD_BITS_TO_STR[c0][RANK] + CARD_BITS_TO_STR[c0][SUITE]]
        c1_1d = dict_str_to_sk[CARD_BITS_TO_STR[c1][RANK] + CARD_BITS_TO_STR[c1][SUITE]]
        board = BOARD_BITS_TO_STR[board_mask.astype(bool)]
        # board = array(['A', 'c', '2', 'h', '8', 'd'], dtype='<U1')
        board_cards = []
        for i in range(0, sum(board_mask) - 1, 2):  # sum is 6,8,10 for flop turn river resp.
            board_cards.append(dict_str_to_sk[board[i] + board[i + 1]])

        return [c0_1d, c1_1d], board_cards

    def look_at_cards(self, obs: np.array) -> Tuple[List[int], List[int]]:
        c0_bits = obs[IDX_C0_0:IDX_C0_1].astype(bool)
        c1_bits = obs[IDX_C1_0:IDX_C1_1].astype(bool)
        board_bits = obs[IDX_BOARD_START:IDX_BOARD_END].astype(int)  # bit representation
        return self.card_bit_mask_to_int(c0_bits, c1_bits, board_bits)

    def _compute_action(self,
                        obs: Union[List, np.ndarray]):
        hero_cards_1d, board_cards_1d = self.look_at_cards(obs)
        # from cards get winning probabilityx
        mc_dict = self._card_evaluator.run_mc(hero_cards_1d, board_cards_1d, 2, n_iter=self._mc_iters)
        # {won: 0, lost: 0, tied: 0}[
        win_prob = float(mc_dict['won'] / self._mc_iters)
        # todo: replace win_prob < .5
        # write script to determine
        # do pseudo harmonic mapping here
        if win_prob < .5 and random() < self.tightness:
            return ActionSpace.FOLD.value
        else:
            assert len(self._predictions == 1)
            prediction = self._predictions[0]
            # return raise of size at most the predicted size bucket
            if min(int(prediction), 2) in self.next_legal_moves:
                return int(prediction)
            elif ActionSpace.CHECK_CALL in self.next_legal_moves:
                return ActionSpace.CHECK_CALL.value
            else:
                return ActionSpace.FOLD.value

    def compute_action(self, obs: np.ndarray, legal_moves):
        self.next_legal_moves = legal_moves
        self._logits = self._model(torch.Tensor(np.array([obs])))
        self._predictions = torch.argmax(self._logits, dim=1)
        action = self._compute_action(obs)
        return action