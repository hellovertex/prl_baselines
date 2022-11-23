import enum
from typing import List, Union, Tuple

import numpy as np
import torch
from prl.environment.Wrappers.prl_wrappers import ActionSpace

from prl.baselines.supervised_learning.models.nn_model import MLP
from prl.baselines.cpp_hand_evaluator.monte_carlo import HandEvaluator_MonteCarlo
from  ray.rllib.policy.policy import Policy

IDX_C0_0 = 167  # feature_names.index('0th_player_card_0_rank_0')
IDX_C0_1 = 184  # feature_names.index('0th_player_card_1_rank_0')
IDX_C1_0 = 184  # feature_names.index('0th_player_card_1_rank_0')
IDX_C1_1 = 201  # feature_names.index('1th_player_card_0_rank_0')
IDX_BOARD_START = 82  #
IDX_BOARD_END = 167  #
N_FEATURES = 564  #
CARD_BITS = np.array(['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c'])
BOARD_BITS = np.array(['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
                       'h', 'd', 's', 'c', '2', '3', '4', '5', '6', '7', '8', '9', 'T',
                       'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c', '2', '3', '4', '5', '6',
                       '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c', '2',
                       '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h',
                       'd', 's', 'c', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J',
                       'Q', 'K', 'A', 'h', 'd', 's', 'c'])
SUITS_HAND_EVALUATOR = ['s', 'h', 'd', 'c']
RANKS_HAND_EVALUATOR = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
CARDS_HAND_EVALUTOR_1D = ['As',
                          'Ah',
                          'Ad',
                          'Ac',
                          'Ks',
                          'Kh',
                          'Kd',
                          'Kc',
                          'Qs',
                          'Qh',
                          'Qd',
                          'Qc',
                          'Js',
                          'Jh',
                          'Jd',
                          'Jc',
                          'Ts',
                          'Th',
                          'Td',
                          'Tc',
                          '9s',
                          '9h',
                          '9d',
                          '9c',
                          '8s',
                          '8h',
                          '8d',
                          '8c',
                          '7s',
                          '7h',
                          '7d',
                          '7c',
                          '6s',
                          '6h',
                          '6d',
                          '6c',
                          '5s',
                          '5h',
                          '5d',
                          '5c',
                          '4s',
                          '4h',
                          '4d',
                          '4c',
                          '3s',
                          '3h',
                          '3d',
                          '3c',
                          '2s',
                          '2h',
                          '2d',
                          '2c']
DICT_CARDS_HAND_EVALUATOR = {'As': 0,
                             'Ah': 1,
                             'Ad': 2,
                             'Ac': 3,
                             'Ks': 4,
                             'Kh': 5,
                             'Kd': 6,
                             'Kc': 7,
                             'Qs': 8,
                             'Qh': 9,
                             'Qd': 10,
                             'Qc': 11,
                             'Js': 12,
                             'Jh': 13,
                             'Jd': 14,
                             'Jc': 15,
                             'Ts': 16,
                             'Th': 17,
                             'Td': 18,
                             'Tc': 19,
                             '9s': 20,
                             '9h': 21,
                             '9d': 22,
                             '9c': 23,
                             '8s': 24,
                             '8h': 25,
                             '8d': 26,
                             '8c': 27,
                             '7s': 28,
                             '7h': 29,
                             '7d': 30,
                             '7c': 31,
                             '6s': 32,
                             '6h': 33,
                             '6d': 34,
                             '6c': 35,
                             '5s': 36,
                             '5h': 37,
                             '5d': 38,
                             '5c': 39,
                             '4s': 40,
                             '4h': 41,
                             '4d': 42,
                             '4c': 43,
                             '3s': 44,
                             '3h': 45,
                             '3d': 46,
                             '3c': 47,
                             '2s': 48,
                             '2h': 49,
                             '2d': 50,
                             '2c': 51}


class AgentModelType(enum.IntEnum):
    MLP_2x512 = 10
    RANDOM_FOREST = 20


class MCPolicy(Policy):
    # todo take rllib policy base and implement custom policy that uses the fold prob sampling method
    # todo give it a reasonable name
    pass


class Agent:
    # todo consider taking rllib agent base
    def __init__(self, env=None):
        # todo build config dict
        self._model = None
        self._card_evaluator = HandEvaluator_MonteCarlo()
        self._policy = None
        self._env = env
        self._feature_names = None
        self._sklansky = 1
        self._initialize()
        self._mc_iters = 5000

    def _initialize(self):
        if self._env:
            self._feature_names = list(self._env.obs_idx_dict.keys()) + ["button_index"]

    def load_model(self, model_type: AgentModelType):
        if model_type == AgentModelType.MLP_2x512:
            input_dim = N_FEATURES
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
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                net = net.cuda()
            self._model = net
            return net
        else:
            raise NotImplementedError

    def card_bit_mask_to_int(self, c0: np.array, c1: np.array, board_mask: np.array) -> Tuple[List[int], List[int]]:
        # todo: docstring and test
        c0_1d = DICT_CARDS_HAND_EVALUATOR[CARD_BITS[c0][0] + CARD_BITS[c0][1]]
        c1_1d = DICT_CARDS_HAND_EVALUATOR[CARD_BITS[c1][0] + CARD_BITS[c1][1]]
        board = BOARD_BITS[board_mask.astype(bool)]

        board_cards = []
        for i in range(0, sum(board_mask) - 1, 2):  # sum is 6,8,10 for flop turn river resp.
            board_cards.append(DICT_CARDS_HAND_EVALUATOR[board[i] + board[i + 1]])

        return [c0_1d, c1_1d], board_cards

    def look_at_cards(self, obs: np.array, feature_names=None) -> Tuple[List[int], List[int]]:
        # todo: docstring and test
        c0 = obs[IDX_C0_0:IDX_C0_1].astype(bool)  # bit representation
        c1 = obs[IDX_C1_0:IDX_C1_1].astype(bool)  # bit representation
        board_mask = obs[IDX_BOARD_START:IDX_BOARD_END].astype(int)  # bit representation
        return self.card_bit_mask_to_int(c0, c1, board_mask)

    def _fold_prob(self, win_prob):
        # todo change later using sklansky values
        return self._sklansky - win_prob
        # return self._sklansky * win_prob

    def act(self, obs: Union[np.array, List]):
        # from obs, get cards
        hero_cards_1d, board_cards_1d = self.look_at_cards(obs, self._feature_names)
        # from cards get winning probability
        mc_dict = self._card_evaluator.run_mc(hero_cards_1d, board_cards_1d, 2, n_iter=self._mc_iters)
        win_prob = float(mc_dict['won'] / self._mc_iters)
        # todo: preflop - n player lookup (equities do)
        #  flop: compute [Flop, 1326] and rank hero vs 1325 => equity estimate
        #  test this using known hand and board combinations
        #  assume Rank(.,.) is quick and correct for now
        # from ranking get fold prob
        fold_prob = self._fold_prob(win_prob)
        # from fold prob get policy and from policy, return action
        action = self._policy.sample(obs, fold_prob)

