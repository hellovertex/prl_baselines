from dataclasses import dataclass
from typing import TypeVar, List, Optional, Union, Dict, Callable, Tuple, Any, Type

from prl.environment.Wrappers.base import EnvWrapperBase
from prl.environment.steinberger.PokerRL import NoLimitHoldem
from ray.rllib import Policy

from prl.baselines.agents.core.base_agent import Agent, RllibAgent
from prl.baselines.supervised_learning.data_acquisition.core.parser import Action

POLICY = TypeVar('POLICY', bound=Policy)
AGENT = TypeVar('AGENT', bound=Agent)
ENV_WRAPPER = TypeVar('ENV_WRAPPER', bound=EnvWrapperBase)

DEFAULT_DATE = "2022-12-24"
DEFAULT_VARIANT = "HUNL"
DEFAULT_CURRENCY = "$"


@dataclass
class PokerExperimentEvaluation:
    pass


@dataclass
class PokerExperimentParticipant:
    """Might change in the future"""
    id: int
    name: str
    alias: Optional[str]
    starting_stack: Union[int, float]
    agent: RllibAgent
    config: Optional[Dict]


@dataclass
class PokerExperiment:
    """Might change in the future"""
    num_players: int  # 2 <= num_players <= 6
    env_cls = NoLimitHoldem
    env: ENV_WRAPPER
    env_reset_config: Optional[Dict[str, Any]]
    starting_stack_sizes: Optional[List[int]]
    # candidates to add
    participants: Optional[Dict[int, PokerExperimentParticipant]]
    # agents: List[AGENT]
    max_episodes: int
    # should PokerExperiments be updated?
    current_episode: Optional[int]
    # callbacks
    cbs_metrics: Optional[List[Callable]]
    cbs_plots: Optional[List[Callable]]
    cbs_misc: Optional[List[Callable]]
    # When no participants are passed, you can use an action plan instead.
    # An action plan contains a single ordered list of actions to execute per episode
    # This means len(from_action_plan) == max_episodes
    from_action_plan: Optional[List[List[Action]]]
