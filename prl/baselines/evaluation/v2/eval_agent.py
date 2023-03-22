import random
from typing import Union, List

import numpy as np
import torch
from tianshou.data import Batch
import numpy as np
from tianshou.policy import BasePolicy


class EvalAgentBase:
    """Wrapper around any agent to match our evaluation interface"""

    def __init__(self, name, *args, **kwargs):
        self.name = name

    def act(self, *args, **kwargs):
        raise NotImplementedError


class EvalAgentRanges(EvalAgentBase):
    def __init__(self, name, base_agent, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self._agent = base_agent

    def act(self, obs: np.ndarray):
        # can easily mask here but no need atm
        act = self._agent(Batch(obs=obs, info={}))
        return act.act[0]


class EvalAgentCall(EvalAgentBase):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def act(self, obs: np.ndarray, legal_moves):
        return 1


class EvalAgentRandom(EvalAgentBase):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def act(self, obs: np.ndarray, legal_moves):
        return random.randint(0, 2)


class EvalAgentRainbow(EvalAgentBase):
    def __init__(self, name, agent):
        super().__init__(name)
        self.agent: BasePolicy = agent

    def act(self, obs, legal_moves):
        obs = dict(obs=obs,
                   mask=legal_moves)
        batch = Batch(obs=obs, info={})
        act = self.agent.forward(batch)
        return act.act[0]



class EvalAgentTianshou(EvalAgentBase):

    def __init__(self, name, agent):
        super().__init__(name)
        self.agent = agent

    def act(self,
            obs: Union[Batch, np.ndarray],
            legal_moves: Union[List, np.ndarray]):
        """
        Dispatches observation(s) --possibly batched-- to wrapped agent.
        If wrapped agent is Tianshou agent, `wrapped_agent.forward` will be called,
        otherwise `wrapped_agent.act` will be called.
        :param obs:
        :return:
        """
        return self.agent.act(obs, legal_moves)
