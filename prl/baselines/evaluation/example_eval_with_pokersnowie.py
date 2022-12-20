from typing import List, Dict

from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env

from prl.baselines.agents.agents import BaselineAgent, CallingStation
from prl.baselines.agents.core.base_agent import Agent, RllibAgent
from prl.baselines.agents.policies import StakeLevelImitationPolicy, AlwaysCallingPolicy
from prl.baselines.evaluation.core.experiment import AGENT, PokerExperiment, PokerExperimentParticipant
from prl.baselines.evaluation.pokersnowie.export import PokerExperimentToPokerSnowie


def make_agents(env, path_to_torch_model_state_dict):
    path_to_baseline_torch_model_state_dict = "/home/sascha/Documents/github.com/prl_baselines/data/ckpt.pt"

    policy_config = {'path_to_torch_model_state_dict': path_to_torch_model_state_dict}
    baseline_policy = StakeLevelImitationPolicy(env.observation_space, env.action_space, policy_config)
    reference_policy = AlwaysCallingPolicy(env.observation_space, env.action_space, policy_config)

    baseline_agent =
    reference_agent = BaselineAgent({'rllib_policy_cls': AlwaysCallingPolicy})
    return [baseline_agent, baseline_agent]


def make_participants(agents: List[RllibAgent], starting_stack_size: int) -> Dict[int, PokerExperimentParticipant]:
    participants = {}
    for i, agent in enumerate(agents):
        participants[i] = PokerExperimentParticipant(id=i,
                                                     name=f'{type(agent.policy)}',
                                                     alias=f'Agent_{i}',
                                                     starting_stack=starting_stack_size,
                                                     agent=agent,
                                                     config={})
    return participants


if __name__ == '__main__':
    # move this to example.py or main.py
    # Construct Experiment
    # todo fix showdown players always only one player
    # todo fix player stack equal to 0
    starting_stack_size = 5000
    num_players = 2
    max_episodes = 100
    env_wrapped = init_wrapped_env(env_wrapper_cls=AugmentObservationWrapper,
                                   stack_sizes=[5000, 5000],
                                   multiply_by=1)

    agents = [
        CallingStation,
        CallingStation
    ]
    participants = make_participants(agents, starting_stack_size)

    experiment = PokerExperiment(
        num_players=num_players,
        env_wrapper_cls=AugmentObservationWrapper,  # single environment to run sequential games on
        starting_stack_size=5000,
        env_config=None,  # can pass {'deck_state_dict': Dict[str, Any]} to init the deck and player cards
        participants=participants,  # wrapper around agents that hold rllib policies that act given observation
        max_episodes=max_episodes,  # number of games to run
        current_episode=0,
        cbs_plots=[],
        cbs_misc=[],
        cbs_metrics=[],
        from_action_plan=None  # compute action from fixed series of actions instead of calls to agent.act
    )
    db_gen = PokerExperimentToPokerSnowie().generate_database(
        path_out='/home/sascha/Documents/github.com/prl_baselines/prl/baselines/evaluation/Pokersnowie2.txt',
        experiment=experiment,
        max_episodes_per_file=500)
