from random import random, randint
from typing import Optional, Any, Dict
from typing import Tuple, List, Union

import numpy as np
import torch
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as cols
from prl.environment.Wrappers.base import ActionSpace
from tianshou.data import Batch
from tianshou.policy import BasePolicy
from tianshou.utils.net.common import MLP
from torch import softmax

from prl.baselines.agents.rule_based import RuleBasedAgent
from prl.baselines.cpp_hand_evaluator.rank import dict_str_to_sk
from prl.baselines.evaluation.utils import pretty_print

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


class TianshouCallingStation(BasePolicy):
    CHECK_CALL = 1

    def __init__(self, observation_space=None, action_space=None):
        super().__init__(observation_space=observation_space,
                         action_space=action_space)

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, **kwargs: Any) -> Batch:
        nobs = len(batch.obs)
        return Batch(logits=None, act=[self.CHECK_CALL] * nobs, state=None)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        return {}


class TianshouRandomAgent(BasePolicy):
    CHECK_CALL = 1

    def __init__(self, observation_space=None, action_space=None):
        super().__init__(observation_space=observation_space,
                         action_space=action_space)

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, **kwargs: Any) -> Batch:
        nobs = len(batch.obs)
        return Batch(logits=None, act=[randint(0, 2)] * nobs,
                     state=None)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        return {}


class MajorityBaseline(BasePolicy):
    def __init__(self,
                 model_ckpt_paths: List[str],
                 model_hidden_dims: List[List[int]],
                 flatten_input=False,
                 device=None,
                 observation_space=None,
                 action_space=None
                 ):
        super().__init__(observation_space=observation_space,
                         action_space=action_space)
        self.model_ckpt_paths = model_ckpt_paths
        self.hidden_dims = model_hidden_dims
        if device is None:
            # todo: time inference times cuda vs cpu
            #  in case obs comes from cpu
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._models = []
        self.load_models(model_ckpt_paths, flatten_input)

    def load_models(self, paths, flatten=False):
        for mpath, hidden_dim in list(zip(paths, self.hidden_dims)):
            self._models.append(self.load_model(mpath, False, hidden_dim))

    def load_model(self, ckpt_path, flatten_input, hidden_dims):
        input_dim = 564
        classes = [ActionSpace.FOLD,
                   ActionSpace.CHECK_CALL,  # CHECK IS INCLUDED in CHECK_CALL
                   ActionSpace.RAISE_MIN_OR_3BB,
                   ActionSpace.RAISE_6_BB,
                   ActionSpace.RAISE_10_BB,
                   ActionSpace.RAISE_20_BB,
                   ActionSpace.RAISE_50_BB,
                   ActionSpace.RAISE_ALL_IN]
        output_dim = len(classes)
        model = MLP(input_dim,
                    output_dim,
                    hidden_dims,
                    flatten_input=flatten_input).to(self.device)
        ckpt = torch.load(ckpt_path,
                          map_location=self.device)
        model.load_state_dict(ckpt['net'])
        model.eval()
        return model

    def compute_action(self, obs: np.ndarray, legal_moves) -> int:
        self.next_legal_moves = legal_moves
        if not type(obs) == torch.Tensor:
            obs = torch.Tensor(np.array([obs]))
        logits = []
        for m in self._models:
            logits.append(m(obs))
        self.logits = torch.mean(torch.stack(logits), dim=0)
        # # in case we ever need the individual predictions:
        # predictions = []
        # for l in logits:
        #     predictions.append(torch.argmax(l, dim=1))
        return torch.argmax(self.logits).item()

    def act(self, obs: np.ndarray, legal_moves: list, use_pseudo_harmonic_mapping=False):
        """
        See "Action translation in extensive-form games with large action spaces:
        Axioms, paradoxes, and the pseudo-harmonic mapping" by Ganzfried and Sandhol

        for why pseudo-harmonic-mapping is useful to prevent exploitability of a strategy.
        """
        self.next_legal_moves = legal_moves
        if not type(obs) == torch.Tensor:
            obs = torch.Tensor(np.array([obs])).to(self.device)
        logits = []
        for m in self._models:
            logits.append(m(obs))
        # predictions = torch.mean(torch.stack(logits), dim=0)
        self.logits = torch.mean(torch.stack(logits), dim=0)
        return torch.argmax(self.logits).item()

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, **kwargs: Any) -> Batch:
        tobs = torch.Tensor(batch.obs['obs']).to(self.device)
        logits = []
        for m in self._models:
            logits.append(m(tobs))
        predictions = torch.mean(torch.stack(logits), dim=0)
        return Batch(logits=None, act=torch.argmax(predictions, dim=1).cpu(), state=None)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        return {}


class BaselineAgent(BasePolicy):
    """ Tianshou Agent -- used with tianshou training"""

    def load_model(self, ckpt_path, flatten_input):
        input_dim = 569
        self._model = MLP(input_dim,
                          3,
                          list(self.hidden_dims),
                          flatten_input=flatten_input).to(self.device)
        ckpt = torch.load(ckpt_path,
                          map_location=self.device)
        self._model.load_state_dict(ckpt['net'])
        self._model.eval()

    def __init__(self,
                 model_ckpt_path: str,
                 flatten_input=False,
                 model_hidden_dims=(256,),
                 device=None,
                 observation_space=None,
                 num_players=None,
                 action_space=None
                 ):
        super().__init__(observation_space=observation_space,
                         action_space=action_space)
        self.model_ckpt_path = model_ckpt_path
        self.hidden_dims = model_hidden_dims
        if device is None:
            # todo: time inference times cuda vs cpu
            #  in case obs comes from cpu
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.load_model(model_ckpt_path, flatten_input)
        self._normalization = None
        self.num_players = num_players
        self.rule_based: Optional[RuleBasedAgent] = None

    @property
    def normalization(self):
        return self._normalization

    @normalization.setter
    def normalization(self, normalization):
        self._normalization = normalization
        self.rule_based = RuleBasedAgent(self.num_players, self._normalization)

    def compute_action(self, obs: np.ndarray, legal_moves) -> int:
        self.next_legal_moves = legal_moves
        if not type(obs) == torch.Tensor:
            obs = torch.Tensor(np.array([obs])).to(self.device)
        self.logits = self._model(obs)
        self.probas = torch.softmax(self.logits, dim=1)
        self.probas = self.probas[:, 1:]
        pred = torch.argmax(self.probas).item() + 1
        proba = torch.max(self.probas)
        threshold_all_in = .18
        threshold_50bb = .18
        threshold_20bb = .16
        threshold_10bb = .14
        threshold_6bb = .15
        threshold_3bb = .16
        threshold_call = .17
        thresholds = [.17, .16, .15, .14, .16, .18, .18]
        if pred == 0:
            return pred
        if proba < thresholds[pred]:
            self._prediction = 0
            return ActionSpace.FOLD
        else:
            self._prediction = pred
            return pred

    def act(self, obs: np.ndarray, legal_moves: list, use_pseudo_harmonic_mapping=False):
        """
        See "Action translation in extensive-form games with large action spaces:
        Axioms, paradoxes, and the pseudo-harmonic mapping" by Ganzfried and Sandhol

        for why pseudo-harmonic-mapping is useful to prevent exploitability of a strategy.
        """
        if obs[0][cols.Round_preflop] == 1:
            return self.rule_based.act(obs[0], legal_moves)

        self.next_legal_moves = legal_moves
        if not type(obs) == torch.Tensor:
            obs = torch.Tensor(np.array([obs])).to(self.device)

        self.logits = self._model(obs)
        self.probas = torch.softmax(self.logits, dim=2)
        pred = torch.argmax(self.probas, dim=2).item()
        proba = torch.max(self.probas, dim=2).values.item()
        threshold_all_in = .18
        threshold_50bb = .18
        threshold_20bb = .16
        threshold_10bb = .14
        threshold_6bb = .15
        threshold_3bb = .16
        threshold_call = .17
        thresholds = [.1, .17, .17, .1593, .1431, .1616, .1815, .1816]

        if proba < thresholds[pred]:
            if self.probas[0][0][0] > thresholds[0]:
                self._prediction = 0
                return 0
            else:
                self._prediction = 1
                return 1
        else:
            self._prediction = pred
            return pred
        # obs = obs[0]
        # self.threshold = .8
        # self.legal_moves = legal_moves
        # self.logits = self._model(torch.Tensor(torch.Tensor(np.array(obs))).to(self.device))
        # self.probas = torch.softmax(self.logits, dim=0)
        # # if this torch.topk(self.logits, 2) is less than 20%
        # # topk = torch.topk(self.logits, 2)
        # # diff = topk.values[0][0] - topk.values[0][1]
        # # thresh = torch.max(topk.values).item() * .2
        # # if diff < thresh:
        # #     # do pseudo-harmonic mapping
        # #     # print('pseudo harmonic mapping')
        # #     pass
        # self._prediction = torch.argmax(self.logits)
        # # if self.threshold <= torch.max(self.probas).detach().cpu().item():
        # pretty_print(99, obs, self._prediction.detach().cpu().item())
        # print(f'Previous action {self._prediction} has been performed with probas {self.probas}')
        # # if self.threshold > torch.max(self.probas).detach().cpu().item():
        # #     return ActionSpace.FOLD
        # return self._prediction.item()

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, **kwargs: Any) -> Batch:
        nobs = len(batch.obs)
        return Batch(logits=None, act=[self.CHECK_CALL] * nobs, state=None)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        return {}
