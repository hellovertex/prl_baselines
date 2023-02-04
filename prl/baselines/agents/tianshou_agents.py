from typing import Union, Optional, Any, Dict

import numpy as np
from tianshou.data import Batch
from tianshou.policy import BasePolicy


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


class BaselineAgent:
    """ Tianshou Agent -- used with tianshou training"""

    def __init__(self, model_ckpt_path: str,
                 observation_space=None,
                 action_space=None):
        super().__init__(observation_space=observation_space,
                         action_space=action_space)
        self.model_ckpt_path = model_ckpt_path

    def load_model(self, path_to_ckpt):
        pass

    def act(self, obs):
        pass

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, **kwargs: Any) -> Batch:
        nobs = len(batch.obs)
        return Batch(logits=None, act=[self.CHECK_CALL] * nobs, state=None)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        return {}
