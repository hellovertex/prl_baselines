from typing import Union

import numpy as np
from tianshou.data import Batch


class EvalAgentBase:
    """Wrapper around any agent to match our evaluation interface"""

    def __init__(self, name, *args, **kwargs):
        self.name = name

    def act(self, *args, **kwargs):
        raise NotImplementedError


class EvalAgent(EvalAgentBase):

    def __init__(self, name, agent):
        super().__init__(name)
        self.agent = agent

    def act(self, obs: Union[Batch, np.ndarray]):
        """
        Dispatches observation(s) --possibly batched-- to wrapped agent.
        If wrapped agent is Tianshou agent, `wrapped_agent.forward` will be called,
        otherwise `wrapped_agent.act` will be called.
        :param obs:
        :return:
        """
        pass
