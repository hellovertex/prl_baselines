from enum import IntEnum
from typing import Union, Optional, Any, Dict

import numpy as np
from tianshou.data import Batch
from tianshou.policy import BasePolicy


class MultiAgentActionFlags(IntEnum):
    """
    In the Baseline Agents, the Monte Carlo simulation takes a lot of time,
    so we want to parallelize its computations.
    Therefore, we include the MC simulator into the environment and let it be parallelized from
    tianshou by triggering the environment to compute the MC-based agent, instead of
    getting it from the agent itself. This way when calling step with an MCTrigger - Action,
    the environment asks the BaselineAgent to compute its action as part of the env.step functionality.
    This parallelizes one MC-simulation per num_env. So we get maximum speedup by
    setting num_envs equal to the number of CPU cores available, e.g. 32.
    And can run 32 MC simulations at the same time -- the same number of env.step() we can call at
    the same time."""
    TriggerMC = 99


class MCPolicy(BasePolicy):
    def __init__(self, observation_space=None, action_space=None):
        super().__init__(observation_space=observation_space,
                         action_space=action_space)

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, **kwargs: Any) -> Batch:
        nobs = len(batch.obs)
        return Batch(logits=None, act=[MultiAgentActionFlags.TriggerMC] * nobs, state=None)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, Any]:
        return {}

