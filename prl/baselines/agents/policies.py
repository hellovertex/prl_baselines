import enum
from random import random
from typing import Union, List, Optional, Dict, Tuple

import numpy as np
import ray
import torch
from prl.environment.Wrappers.base import ActionSpace
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as FeatureEnum
from ray.rllib.evaluation import Episode
from ray.rllib.utils.typing import TensorStructType, TensorType

from prl.baselines.agents.core.base_policy import BaselinePolicy_Base
from prl.baselines.cpp_hand_evaluator.monte_carlo import HandEvaluator_MonteCarlo
from prl.baselines.supervised_learning.models.nn_model import MLP
from prl.baselines.cpp_hand_evaluator.rank import dict_str_to_sk

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


class RandomPokerPolicy(BaselinePolicy_Base):
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


class AlwaysCallingPolicy(BaselinePolicy_Base):
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
        return [torch.tensor(1) for _ in obs_batch], [], {}


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
        return [torch.tensor(2) for _ in obs_batch], [], {}


class BaselineModelType(enum.IntEnum):
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

    def __init__(self,
                 observation_space,
                 action_space,
                 config,
                 tightness: float = 0.9,
                 acceptance_level: float = 0.7):
        BaselinePolicy_Base.__init__(self, observation_space, action_space, config)
        self._logits = None
        self._path_to_torch_model_state_dict = config['path_to_torch_model_state_dict']
        self._model = self.load_model()
        self.tightness = tightness
        self.acceptance_level = acceptance_level
        self._card_evaluator = HandEvaluator_MonteCarlo()
        self._mc_iters = 5000
        self._predictions = []
        self._legal_moves = []

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

    def compute_action(self,
                       agent_id: int,
                       obs: Union[List, np.ndarray]):
        hero_cards_1d, board_cards_1d = self.look_at_cards(obs)
        # from cards get winning probabilityx
        mc_dict = self._card_evaluator.run_mc(hero_cards_1d, board_cards_1d, 2, n_iter=self._mc_iters)
        # {won: 0, lost: 0, tied: 0}[
        win_prob = float(mc_dict['won'] / self._mc_iters)
        if win_prob < .5 and random() < self.tightness:
            return torch.tensor(ActionSpace.FOLD.value)
        else:
            prediction = self._predictions[agent_id]
            # return raise of size at most the predicted size bucket
            if min(int(prediction), 2) in self._legal_moves[agent_id]:
                return prediction
            elif ActionSpace.CHECK_CALL in self._legal_moves[agent_id]:
                return torch.tensor(ActionSpace.CHECK_CALL.value)
            else:
                return torch.tensor(ActionSpace.FOLD.value)

    def compute_actions(self,
                        obs_batch: Union[List[TensorStructType], TensorStructType],
                        state_batches: Optional[List[TensorType]] = None,
                        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
                        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
                        info_batch: Optional[Dict[str, list]] = None,
                        episodes: Optional[List["Episode"]] = None,
                        explore: Optional[bool] = None,
                        timestep: Optional[int] = None,
                        **kwargs, ):
        # obs_batch = ndarray(n_envs, len(observation_space))
        # 1. todo consider canCheck, canRaise, canBet, canCall, canAllIn vs legalMoves
        # 2.  [Optional] try change strategy
        # 3. if EHS < .5 and random(0,1) > tightness: fold
        # 4. if acceptance_level < max(model(obs)): return argmax(model(obs)))
        # 5. else: return fold or return max(fold*tightness, max(model(obs)))
        if not (type(obs_batch) == dict or type(obs_batch[0]) == dict):
            raise NotImplementedError("Observation should be dictionary, i.e."
                                      "containing 'legal_moves' mask. Make sure to disable rllib preprocessor"
                                      "api in the environment config to avoid auto-flattening observation.")
        # the legal moves are the first three bits of each observation, as of rllib v2.1.0
        # Tuple and Dict observations are flattened by the preprocessor
        # we postpone implementing our own and use this implicit conversion for now
        self._legal_moves = obs_batch['legal_moves']
        # extract legal move bits to obtain original observations
        observations = obs_batch['obs']
        self._logits = self._model(torch.Tensor(observations))
        self._predictions = torch.argmax(self._logits, axis=1)
        return [self.compute_action(idx, obs) for idx, obs in enumerate(observations)], [], {}

    def load_model(self, model_type: BaselineModelType = BaselineModelType.MLP_2x512):
        if model_type == BaselineModelType.MLP_2x512:
            input_dim = len(FeatureEnum)
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
            self._model.load_state_dict(torch.load(self._path_to_torch_model_state_dict,
                                                   # always on cpu because model used to collects rollouts
                                                   map_location=torch.device('cpu'))['net'])
            self._model.eval()
            return net
        else:
            raise NotImplementedError

try:
    ray.rllib.algorithms.registry.POLICIES[
        'StakeLevelImitationPolicy'] = 'prl.baselines.agents.policies.StakeLevelImitationPolicy'
    ray.rllib.algorithms.registry.POLICIES[
        'AlwaysCallingPolicy'] = 'prl.baselines.agents.policies.AlwaysCallingPolicy'
    ray.rllib.algorithms.registry.POLICIES[
        'RandomPokerPolicy'] = 'prl.baselines.agents.policies.RandomPokerPolicy'
except AttributeError as e:
    pass