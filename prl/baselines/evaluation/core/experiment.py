import enum
from dataclasses import dataclass
from typing import TypeVar, List, Optional, Union, Dict, Callable, Tuple, Any, Type

import tianshou.policy
from prl.environment.Wrappers.base import EnvWrapperBase
from prl.environment.steinberger.PokerRL import NoLimitHoldem
from ray.rllib import Policy
from tianshou.env import SubprocVectorEnv

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
    agent: tianshou.policy.BasePolicy
    config: Optional[Dict]


def make_participants(agents, starting_stack, **kwargs) -> Tuple[PokerExperimentParticipant]:
    """Returns list of agents using default participant represenation"""
    participants = []
    for i, agent in enumerate(agents):
        participants.append(PokerExperimentParticipant(id=i,
                                                       name=f'{type(agent).__name__}_Seat_{i + 1}',
                                                       alias=f'Agent_{i}',
                                                       starting_stack=starting_stack,
                                                       agent=agent,
                                                       config={}))
    return tuple(participants)


class PokerExperiment_EarlyStopping(enum.IntEnum):
    PLAY_UNTIL_FIRST_ELIMINATED_PLAYER: 0
    PLAY_UNTIL_HEADS_UP_WON: 1
    ALWAYS_REBUY_AND_PLAY_UNTIL_NUM_EPISODES_REACHED: 99


@dataclass
class PokerExperiment:
    """Might change in the future"""
    num_players: int  # 2 <= num_players <= 6
    env_cls = NoLimitHoldem
    wrapped_env: ENV_WRAPPER
    env_reset_config: Optional[Dict[str, Any]]
    starting_stack_sizes: Optional[List[int]]
    # candidates to add
    participants: Optional[Tuple[PokerExperimentParticipant]]
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
    options: Optional[Dict] = None
    early_stopping_when: Optional[PokerExperiment_EarlyStopping] = None


def make_default_experiment(env, participants, max_episodes=10, env_reset_config=None):
    return PokerExperiment(
        # env
        num_players=len(participants),
        wrapped_env=env,  # single environment to run sequential games on
        env_reset_config=env_reset_config,
        starting_stack_sizes=None,
        # can pass {'deck_state_dict': Dict[str, Any]} to init the deck and player cards
        # run
        max_episodes=max_episodes,  # number of games to run
        current_episode=0,
        cbs_plots=[],
        cbs_misc=[],
        cbs_metrics=[],
        # actors
        participants=participants,  # wrapper around agents that hold rllib policies that act given observation
        from_action_plan=None,  # compute action from fixed series of actions instead of calls to agent.act
        # early_stopping_when=PokerExperiment_EarlyStopping.ALWAYS_REBUY_AND_PLAY_UNTIL_NUM_EPISODES_REACHED
    )
