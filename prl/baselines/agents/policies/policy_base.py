from typing import Union, List, Optional, Dict, Tuple

from ray.rllib import Policy, SampleBatch
from ray.rllib.evaluation import Episode
from ray.rllib.models import ModelV2, ActionDistribution
from ray.rllib.utils.typing import TensorStructType, TensorType, ModelGradients


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


class CallingStation(BaselinePolicy_Base):
    """Policy that always calls"""

    def compute_actions(self, obs_batch: Union[List[TensorStructType], TensorStructType],
                        state_batches: Optional[List[TensorType]] = None,
                        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
                        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
                        info_batch: Optional[Dict[str, list]] = None,
                        episodes: Optional[List["Episode"]] = None,
                        explore: Optional[bool] = None,
                        timestep: Optional[int] = None,
                        **kwargs, ):
        # todo implement
        pass


class StakeLevelImitationPolicy(BaselinePolicy_Base):
    """Uses a pytorch model that was trained in supervised learning regime from real
    online game logs to immitate  set of players. The pytorch model was trained for
    games where the players hand cards were shown, i.e. games that went to showdown
    and finished. The Fold-actions could not be trained, because no labels
    (hand cards) were available to do so. The fold-actions must be heuristically
    determined, and the policies sampling mechanism must be adjusted according,
    i.e. the probability distribution must be re-normalized after determining the
    fold-probability. The heuristic used, to determine the fold-probability, is
    taken from the paper "Building a No Limit Texas Hold'em Poker Agent Based on
    Game Logs Using Supervised Learning". It computes the effective hand-strength
    (EHS) and folds depndending on the tightness level of the baseline agent:
    If EHS < 0.5, the agent has a probability of folding, equal to its tightness
    level.
    """

    def __init__(self, observation_space, action_space, config, tightness: float = 0.8, acceptance_level: float = 0.7):
        BaselinePolicy_Base.__init__(self, observation_space, action_space, config)
        self.tightness = tightness
        self.acceptance_level = acceptance_level

    def compute_actions(self, obs_batch: Union[List[TensorStructType], TensorStructType],
                        state_batches: Optional[List[TensorType]] = None,
                        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
                        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
                        info_batch: Optional[Dict[str, list]] = None,
                        episodes: Optional[List["Episode"]] = None,
                        explore: Optional[bool] = None,
                        timestep: Optional[int] = None,
                        **kwargs, ):
        # 0 todo implement 1-5:
        # 1. canCheck, canRaise, canBet, canCall, canAllIn
        # 2. [Optional] try change strategy
        # 3. if EHS < .5 and random(0,1) > tightness: fold
        # 4. if acceptance_level < max(model(obs)): return argmax(model(obs)))
        # 5. else: return fold or return max(fold*tightness, max(model(obs)))
        return [self.action_space.sample() for _ in obs_batch], [], {}
