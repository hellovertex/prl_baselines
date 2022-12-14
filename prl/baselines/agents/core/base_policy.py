from typing import Union, List, Optional, Dict

from ray.rllib import Policy
from ray.rllib.evaluation import Episode
from ray.rllib.utils.typing import TensorStructType, TensorType


class BaselinePolicy_Base(Policy):
    """BaselinePolicy base class to be overwritten only by Baseline-policies,
    not by RL-policies. BaselinePolicies do not update during RL-training or eval.
    They must only implement the computeActions-fn."""

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        # example parameter
        self.w = 1.0

    def compute_actions(self, obs_batch: Union[List[TensorStructType], TensorStructType],
                        state_batches: Optional[List[TensorType]] = None,
                        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
                        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
                        info_batch: Optional[Dict[str, list]] = None,
                        episodes: Optional[List["Episode"]] = None,
                        explore: Optional[bool] = None,
                        timestep: Optional[int] = None,
                        **kwargs, ):
        """ Implement this in Derived classes. See
        https://docs.ray.io/en/latest/rllib/rllib-concepts.html#policies-in-multi-agent"""
        raise NotImplementedError

    def learn_on_batch(self, samples):
        """Only for consistency with Rllib policy interface """
        # we don't train baseline policies
        pass

    def get_weights(self):
        return {"w": self.w}

    def set_weights(self, weights):
        self.w = weights["w"]
