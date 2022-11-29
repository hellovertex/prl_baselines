import enum
from random import random
from typing import Union, List, Optional, Dict, Tuple

import numpy as np
import torch
from prl.environment.Wrappers.base import ActionSpace
from ray.rllib.evaluation import Episode
from ray.rllib.utils.typing import TensorStructType, TensorType

from prl.baselines.agents.core.policies.policy_base import BaselinePolicy_Base
from prl.baselines.cpp_hand_evaluator.monte_carlo import HandEvaluator_MonteCarlo
from prl.baselines.supervised_learning.models.nn_model import MLP


class CallingStation(BaselinePolicy_Base):
    """Policy that always calls"""

    def compute_actions(self, obs_batch: Union[List[TensorStructType], TensorStructType],
                        state_batches: Optional[List[TensorType]] = None,
                        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
                        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
                        info_batch: Optional[Dict[str, list]] = None,
                        episodes: Optional[List["Episode"]] = None,
                        explore: Optional[bool] = None,
                        timestep: Optional[int] = None,
                        **kwargs, ):
        return [np.int64(1) for _ in obs_batch], [], {}


class AlwaysMinRaise(BaselinePolicy_Base):
    """Policy that always min-raises"""

    def compute_actions(self, obs_batch: Union[List[TensorStructType], TensorStructType],
                        state_batches: Optional[List[TensorType]] = None,
                        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
                        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
                        info_batch: Optional[Dict[str, list]] = None,
                        episodes: Optional[List["Episode"]] = None,
                        explore: Optional[bool] = None,
                        timestep: Optional[int] = None,
                        **kwargs, ):
        return [np.int64(2) for _ in obs_batch], [], {}


class RandomPolicy(BaselinePolicy_Base):
    """Policy that returns Random Actions"""

    def compute_actions(self, obs_batch: Union[List[TensorStructType], TensorStructType],
                        state_batches: Optional[List[TensorType]] = None,
                        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
                        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
                        info_batch: Optional[Dict[str, list]] = None,
                        episodes: Optional[List["Episode"]] = None,
                        explore: Optional[bool] = None,
                        timestep: Optional[int] = None,
                        **kwargs, ):
        return [self.action_space.sample() for _ in obs_batch], [], {}


IDX_C0_0 = 167  # feature_names.index('0th_player_card_0_rank_0')
IDX_C0_1 = 184  # feature_names.index('0th_player_card_1_rank_0')
IDX_C1_0 = 184  # feature_names.index('0th_player_card_1_rank_0')
IDX_C1_1 = 201  # feature_names.index('1th_player_card_0_rank_0')
IDX_BOARD_START = 82  #
IDX_BOARD_END = 167  #
CARD_BITS = np.array(['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c'])
BOARD_BITS = np.array(['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
                       'h', 'd', 's', 'c', '2', '3', '4', '5', '6', '7', '8', '9', 'T',
                       'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c', '2', '3', '4', '5', '6',
                       '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c', '2',
                       '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h',
                       'd', 's', 'c', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J',
                       'Q', 'K', 'A', 'h', 'd', 's', 'c'])
DICT_CARDS_HAND_EVALUATOR = {'As': 0, 'Ah': 1, 'Ad': 2, 'Ac': 3, 'Ks': 4, 'Kh': 5, 'Kd': 6, 'Kc': 7, 'Qs': 8, 'Qh': 9,
                             'Qd': 10, 'Qc': 11, 'Js': 12, 'Jh': 13, 'Jd': 14, 'Jc': 15, 'Ts': 16, 'Th': 17, 'Td': 18,
                             'Tc': 19, '9s': 20, '9h': 21, '9d': 22, '9c': 23, '8s': 24, '8h': 25, '8d': 26, '8c': 27,
                             '7s': 28, '7h': 29, '7d': 30, '7c': 31, '6s': 32, '6h': 33, '6d': 34, '6c': 35, '5s': 36,
                             '5h': 37, '5d': 38, '5c': 39, '4s': 40, '4h': 41, '4d': 42, '4c': 43, '3s': 44, '3h': 45,
                             '3d': 46, '3c': 47, '2s': 48, '2h': 49, '2d': 50, '2c': 51}


class ModelType(enum.IntEnum):
    MLP_2x512 = 10
    RANDOM_FOREST = 20


class StakeLevelImitationPolicy(BaselinePolicy_Base):
    """Uses a pytorch model that was trained in supervised learning regime from real
    online game logs to immitate  set of players. The pytorch model was trained for
    games where the players hand cards were shown, i.e. games that went to showdown
    and finished. The Fold-actions could not be trained, because no labels
    (hand cards) were available to do so. The fold-actions must be heuristically
    determined, and the policies sampling mechanism must be adjusted according,
    i.e. the probability distribution must be re-normalized after determining the
    fold-probability. The heuristic used, to determine the fold-probability, is
    taken from the paper "Building a No Limit Texas Hold'em Poker Agent Based on
    Game Logs Using Supervised Learning". It computes the effective hand-strength
    (EHS) and folds depndending on the tightness level of the baseline agent:
    If EHS < 0.5, the agent has a probability of folding, equal to its tightness
    level.
    """

    def __init__(self, observation_space, action_space, config, tightness: float = 0.8, acceptance_level: float = 0.7):
        BaselinePolicy_Base.__init__(self, observation_space, action_space, config)
        self.tightness = tightness
        self.acceptance_level = acceptance_level
        self._card_evaluator = HandEvaluator_MonteCarlo()
        self._mc_iters = 5000

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

    def compute_action(self, obs: Union[List, np.ndarray]):
        hero_cards_1d, board_cards_1d = self.look_at_cards(obs, self._feature_names)
        # from cards get winning probability
        mc_dict = self._card_evaluator.run_mc(hero_cards_1d, board_cards_1d, 2, n_iter=self._mc_iters)
        # {won: 0, lost: 0, tied: 0}
        win_prob = float(mc_dict['won'] / self._mc_iters)
        if win_prob < .5 and random() > self.tightness:
            return 0
        else:
            # todo query model (finish self.load_model fn) and check if is plausible
            return 1

    def compute_actions(self, obs_batch: Union[List[TensorStructType], TensorStructType],
                        state_batches: Optional[List[TensorType]] = None,
                        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
                        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
                        info_batch: Optional[Dict[str, list]] = None,
                        episodes: Optional[List["Episode"]] = None,
                        explore: Optional[bool] = None,
                        timestep: Optional[int] = None,
                        **kwargs, ):
        # obs_batch = ndarray(n_envs, len(observation_space))
        # 0 todo implement 1-5:
        # 1. canCheck, canRaise, canBet, canCall, canAllIn
        # 2. [Optional] try change strategy
        # 3. if EHS < .5 and random(0,1) > tightness: fold
        # 4. if acceptance_level < max(model(obs)): return argmax(model(obs)))
        # 5. else: return fold or return max(fold*tightness, max(model(obs)))
        return [self.compute_action(obs) for obs in obs_batch], [], {}

    def load_model(self, model_type: ModelType):
        if model_type == ModelType.MLP_2x512:
            input_dim = N_FEATURES = 564
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