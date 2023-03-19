from typing import Union, List

import numpy as np
from tianshou.data import Batch
import numpy as np

class EvalAgentBase:
    """Wrapper around any agent to match our evaluation interface"""

    def __init__(self, name, *args, **kwargs):
        self.name = name

    def act(self, *args, **kwargs):
        raise NotImplementedError

class EvalAgentRanges(EvalAgentBase):
    def __init__(self, base_agent):
        self._agent = base_agent
    def act(obs: np.ndarray):
        pass

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
        return self.agent.act(obs)
