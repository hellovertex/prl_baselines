import os
import time
from functools import partial
from multiprocessing.pool import Pool
from typing import Optional, Union, Any, Dict

import numpy as np
from tianshou.data import Collector, Batch
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, BasePolicy

from prl.baselines.cpp_hand_evaluator.monte_carlo import HandEvaluator_MonteCarlo


class RandomPolicy(BasePolicy):
    """A random agent used in multi-agent learning.

    It randomly chooses an action from the legal action.
    """

    def __init__(self):
        super().__init__()
        self.heval = HandEvaluator_MonteCarlo()
        self.n_cpus = os.cpu_count()
    def fn(self, obs):
        hero_cards, board_cards = [0, 1], []  # todo compute from observation
        return self.heval.run_mc(hero_cards, board_cards, 2, n_iter=5000)
    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            **kwargs: Any,
    ) -> Batch:
        """Compute the random action over the given batch data.

        The input should contain a mask in batch.obs, with "True" to be
        available and "False" to be unavailable. For example,
        ``batch.obs.mask == np.array([[False, True, False]])`` means with batch
        size 1, action "1" is available but action "0" and "2" are unavailable.

        :return: A :class:`~tianshou.data.Batch` with "act" key, containing
            the random action.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        mask = batch.obs.mask
        # hero_cards_1d, board_cards_1d = self.look_at_cards([0 for _ in range(564)])
        fn = partial(self.heval.run_mc, [0, 1], [], 2, n_iter=5000)
        results = []
        # run fn using imapordered
        t0 = time.time()
        with Pool(processes=self.n_cpus-1) as pool:
            for stats in pool.imap(self.fn, [() for _ in range(self.n_cpus-1)]):  # use starmap to call fn without args
                results.append(stats)
        print(f'Monte Carlo Simulation took {time.time() - t0} seconds.\nResults are {results}')
        logits = np.random.rand(*mask.shape)
        logits[~mask] = -np.inf
        return Batch(act=logits.argmax(axis=-1))

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        """Since a random agent learns nothing, it returns an empty dict."""
        return {}


from pettingzoo.classic import rps_v2

if __name__ == "__main__":
    # Step 1: Load the PettingZoo environment
    env = rps_v2.env(render_mode="human")

    # Step 2: Wrap the environment for Tianshou interfacing
    env = PettingZooEnv(env)

    # Step 3: Define policies for each agent
    policies = MultiAgentPolicyManager([RandomPolicy(), RandomPolicy()], env)

    # Step 4: Convert the env to vector format
    env = DummyVectorEnv([lambda: env for _ in range(os.cpu_count()-1)])

    # Step 5: Construct the Collector, which interfaces the policies with the vectorised environment
    collector = Collector(policies, env)

    # Step 6: Execute the environment with the agents playing for 1 episode, and render a frame every 0.1 seconds
    result = collector.collect(n_episode=1, render=0.1)
