from typing import NamedTuple, TypeVar, List, Optional, Union, Dict, Callable
from dataclasses import dataclass
from prl.environment.Wrappers.base import EnvWrapperBase
from prl.environment.steinberger.PokerRL import NoLimitHoldem
from ray.rllib import Policy

from prl.baselines.agents.core.base_agent import Agent

POLICY = TypeVar('POLICY', bound=Policy)
AGENT = TypeVar('AGENT', bound=Agent)
ENV_WRAPPER = TypeVar('ENV_WRAPPER', bound=EnvWrapperBase)


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
    env: Union[ENV_WRAPPER, NoLimitHoldem]
    # candidates to add
    participants: Dict[int, PokerExperimentParticipant]
    max_episodes: int
    agents: List[AGENT]  # wrapper for POLICY
    # should PokerExperiments be updated?
    current_episode: Optional[int]
    # callbacks
    cbs_metrics: Optional[List[Callable]]
    cbs_plots: Optional[List[Callable]]
    cbs_misc: Optional[List[Callable]]
