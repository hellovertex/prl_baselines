from enum import IntEnum
from typing import Union, Optional, Any, Dict

import numpy as np
import torch
from prl.environment.Wrappers.base import ActionSpace
from tianshou.data import Batch
from tianshou.policy import BasePolicy

from prl.baselines.examples.rainbow_net import Rainbow


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


default_rainbow_params = {'device': "cuda",
                          'lr': 1e-6,
                          'num_atoms': 51,
                          'noisy_std': 0.1,
                          'v_min': -6,
                          'v_max': 6,
                          'estimation_step': 3,
                          'target_update_freq': 500  # training steps
                          }


def get_rainbow_config(params):
    # network
    classes = [ActionSpace.FOLD,
               ActionSpace.CHECK_CALL,  # CHECK IS INCLUDED in CHECK_CALL
               ActionSpace.RAISE_MIN_OR_3BB,
               ActionSpace.RAISE_6_BB,
               ActionSpace.RAISE_10_BB,
               ActionSpace.RAISE_20_BB,
               ActionSpace.RAISE_50_BB,
               ActionSpace.RAISE_ALL_IN]
    hidden_dim = [512, 512]
    output_dim = len(classes)
    input_dim = 564  # hard coded for now -- very unlikely to be changed by me at any poiny in time
    device = params['device']
    net = Rainbow(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_dim,
        device=device,
        num_atoms=params['num_atoms'],
        noisy_std=params['noisy_std'],
        is_dueling=True,
        is_noisy=True
    )
    # load from config if possible
    optim = torch.optim.Adam(net.parameters(), lr=params['lr'])
    if 'load_from_ckpt' in params:
        # rainbow_policy.load_state_dict(os.path.join(
        #     *logdir, f'policy_{0}.pth'
        # ))
        try:
            net.load_state_dict(torch.load(params['load_from_ckpt'], map_location=device)['model'])
            optim.load_state_dict(torch.load(params['load_from_ckpt'], map_location=device)['optim'])
        except FileNotFoundError:
            # initial state, no checkpoints created yet, ignore silently
            pass
    # if running on GPU and we want to use cuda move model there
    return {'model': net,
            'optim': optim,
            'num_atoms': params['num_atoms'],
            'v_min': params['v_min'],
            'v_max': params['v_max'],
            'estimation_step': params['estimation_step'],
            'target_update_freq': params['target_update_freq']  # training steps
            }
