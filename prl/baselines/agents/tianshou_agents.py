from random import random
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


class MajorityBaseline(BasePolicy):
    def __init__(self,
                 model_ckpt_paths: List[str],
                 num_players,
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
        self.num_players = num_players
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
                   ActionSpace.RAISE_HALF_POT,
                   ActionSpace.RAISE_POT,
                   ActionSpace.ALL_IN]
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
        predictions = []
        for l in logits:
            predictions.append(torch.argmax(l, dim=1))

        return 1

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
        predictions = torch.mean(torch.stack(logits), dim=0)
        return torch.argmax(predictions).item()

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, **kwargs: Any) -> Batch:
        nobs = len(batch.obs)
        return Batch(logits=None, act=[self.CHECK_CALL] * nobs, state=None)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        return {}


class BaselineAgent(BasePolicy):
    """ Tianshou Agent -- used with tianshou training"""

    def __init__(self,
                 model_ckpt_path: str,
                 num_players,
                 flatten_input=False,
                 model_hidden_dims=(256,),
                 device=None,
                 observation_space=None,
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
        self.num_players = num_players
        self.load_model(model_ckpt_path, flatten_input)

    def load_model(self, ckpt_path, flatten_input):
        input_dim = 564
        classes = [ActionSpace.FOLD,
                   ActionSpace.CHECK_CALL,  # CHECK IS INCLUDED in CHECK_CALL
                   ActionSpace.RAISE_MIN_OR_3BB,
                   ActionSpace.RAISE_HALF_POT,
                   ActionSpace.RAISE_POT,
                   ActionSpace.ALL_IN]
        output_dim = len(classes)
        self._model = MLP(input_dim,
                          output_dim,
                          list(self.hidden_dims),
                          flatten_input=flatten_input).to(self.device)
        ckpt = torch.load(ckpt_path,
                          map_location=self.device)
        self._model.load_state_dict(ckpt['net'])
        self._model.eval()

    def compute_action(self, obs: np.ndarray, legal_moves) -> int:
        self.next_legal_moves = legal_moves
        if not type(obs) == torch.Tensor:
            obs = torch.Tensor(np.array([obs]))
        self._logits = self._model(obs)
        self._predictions = torch.argmax(self._logits, dim=1)
        action = self._compute_action(obs)
        return action

    def act(self, obs: np.ndarray, legal_moves: list, use_pseudo_harmonic_mapping=False):
        """
        See "Action translation in extensive-form games with large action spaces:
        Axioms, paradoxes, and the pseudo-harmonic mapping" by Ganzfried and Sandhol

        for why pseudo-harmonic-mapping is useful to prevent exploitability of a strategy.
        """
        self.legal_moves = legal_moves
        self._logits = self._model(torch.Tensor(torch.Tensor(np.array(obs))).to(self.device))
        # if this torch.topk(self._logits, 2) is less than 20%
        topk = torch.topk(self._logits, 2)
        diff = topk.values[0][0] - topk.values[0][1]
        thresh = torch.max(topk.values).item() * .2
        if diff < thresh:
            # do pseudo-harmonic mapping
            # print('pseudo harmonic mapping')
            pass
        self._prediction = torch.argmax(self._logits)
        return self._prediction.item()

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, **kwargs: Any) -> Batch:
        nobs = len(batch.obs)
        return Batch(logits=None, act=[self.CHECK_CALL] * nobs, state=None)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        return {}
