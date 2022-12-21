from typing import List, Dict, Type, Tuple, Any

import gym
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.base import EnvWrapperBase
from prl.environment.Wrappers.utils import init_wrapped_env

from prl.baselines.agents.agents import CallingStation
from prl.baselines.agents.core.base_agent import Agent, RllibAgent
from prl.baselines.agents.policies import StakeLevelImitationPolicy, AlwaysCallingPolicy
from prl.baselines.evaluation.core.experiment import AGENT, PokerExperiment, PokerExperimentParticipant
from prl.baselines.evaluation.pokersnowie.export import PokerExperimentToPokerSnowie

# def make_agents(env, path_to_torch_model_state_dict):
#     path_to_baseline_torch_model_state_dict = "/home/sascha/Documents/github.com/prl_baselines/data/ckpt.pt"
#
#     policy_config = {'path_to_torch_model_state_dict': path_to_torch_model_state_dict}
#     baseline_policy = StakeLevelImitationPolicy(env.observation_space, env.action_space, policy_config)
#     reference_policy = AlwaysCallingPolicy(env.observation_space, env.action_space, policy_config)
#
#     baseline_agent =
#     reference_agent = BaselineAgent({'rllib_policy_cls': AlwaysCallingPolicy})
#     return [baseline_agent, baseline_agent]

AGENT_CLS = Type[RllibAgent]
POLICY_CONFIG = Dict[str, Any]
STARTING_STACK = int
AGENT_INIT_COMPONENTS = Tuple[AGENT_CLS, POLICY_CONFIG, STARTING_STACK]


def make_participants(agent_init_components: List[AGENT_INIT_COMPONENTS],
                      observation_space: gym.Space,
                      action_space: gym.Space) -> Dict[int, PokerExperimentParticipant]:
    participants = {}
    # todo
    for i, (agent_cls, policy_config, stack) in enumerate(agent_init_components):
        agent_config = {'observation_space': observation_space,
                        'action_space': action_space,
                        'policy_config': policy_config}
        agent = agent_cls(agent_config)
        participants[i] = PokerExperimentParticipant(id=i,
                                                     name=f'{agent_cls.__name__}',
                                                     alias=f'Agent_{i}',
                                                     starting_stack=stack,
                                                     agent=agent,
                                                     config={})
    return participants


if __name__ == '__main__':
    starting_stack_size = 1000
    num_players = 2
    max_episodes = 100
    env = init_wrapped_env(env_wrapper_cls=AugmentObservationWrapper,
                           stack_sizes=[starting_stack_size, starting_stack_size],
                           multiply_by=1)

    agent_init_components = [
        (CallingStation, {}, starting_stack_size),  # agent_cls, policy_config, stack
        (CallingStation, {}, starting_stack_size)  # agent_cls, policy_config, stack
    ]
    participants = make_participants(agent_init_components,
                                     observation_space=env.observation_space,
                                     action_space=env.action_space)

    experiment = PokerExperiment(
        # env
        env=env,  # single environment to run sequential games on
        num_players=num_players,
        starting_stack_sizes=[starting_stack_size, starting_stack_size],
        env_reset_config=None,  # can pass {'deck_state_dict': Dict[str, Any]} to init the deck and player cards
        # run
        max_episodes=max_episodes,  # number of games to run
        current_episode=0,
        cbs_plots=[],
        cbs_misc=[],
        cbs_metrics=[],
        # actors
        participants=participants,  # wrapper around agents that hold rllib policies that act given observation
        from_action_plan=None  # compute action from fixed series of actions instead of calls to agent.act
    )
    db_gen = PokerExperimentToPokerSnowie().generate_database(
        path_out='/home/sascha/Documents/github.com/prl_baselines/prl/baselines/evaluation/Pokersnowie2.txt',
        experiment=experiment,
        max_episodes_per_file=500)
