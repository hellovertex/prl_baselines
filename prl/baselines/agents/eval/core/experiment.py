from dataclasses import dataclass
from typing import TypeVar, List, Optional, Union, Dict, Callable, Tuple, Any

from prl.environment.Wrappers.base import EnvWrapperBase
from prl.environment.steinberger.PokerRL import NoLimitHoldem
from ray.rllib import Policy

from prl.baselines.agents.core.base_agent import Agent
from prl.baselines.supervised_learning.data_acquisition.core.parser import Action

POLICY = TypeVar('POLICY', bound=Policy)
AGENT = TypeVar('AGENT', bound=Agent)
ENV_WRAPPER = TypeVar('ENV_WRAPPER', bound=EnvWrapperBase)


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
    agent: AGENT
    config: Optional[Dict]


@dataclass
class PokerExperiment:
    """Might change in the future"""
    num_players: int  # 2 <= num_players <= 6
    env: Union[ENV_WRAPPER, NoLimitHoldem]
    env_config: Optional[Dict[str, Any]]
    # candidates to add
    participants: Dict[int, PokerExperimentParticipant]
    max_episodes: int
    # should PokerExperiments be updated?
    current_episode: Optional[int]
    # callbacks
    cbs_metrics: Optional[List[Callable]]
    cbs_plots: Optional[List[Callable]]
    cbs_misc: Optional[List[Callable]]
    # from_state_dict: Optional[Dict]  # if hand cards and board cards should be predetermined -- is set via env
    from_action_plan: Optional[List[Union[Action, Tuple]]]  # if actions should be predetermined (e.g. during testing)
