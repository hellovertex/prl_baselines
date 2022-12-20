from typing import List, Dict

from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env

from prl.baselines.agents.agents import BaselineAgent
from prl.baselines.agents.policies import StakeLevelImitationPolicy, CallingStation
from prl.baselines.evaluation.core.experiment import AGENT, PokerExperimentParticipant, PokerExperiment
from prl.baselines.evaluation.pokersnowie.export import PokerExperimentToPokerSnowie


def make_agents(env, path_to_torch_model_state_dict):
    policy_config = {'path_to_torch_model_state_dict': path_to_torch_model_state_dict}
    baseline_policy = StakeLevelImitationPolicy(env.observation_space, env.action_space, policy_config)
    reference_policy = CallingStation(env.observation_space, env.action_space, policy_config)

    baseline_agent = BaselineAgent({'rllib_policy': baseline_policy})
    reference_agent = BaselineAgent({'rllib_policy': reference_policy})
    return [baseline_agent, baseline_agent]


def make_participants(agents: List[AGENT], starting_stack_size: int) -> Dict[int, PokerExperimentParticipant]:
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
    stacks = [starting_stack_size for _ in range(num_players)]
    env_wrapped = init_wrapped_env(env_wrapper_cls=AugmentObservationWrapper,
                                   stack_sizes=stacks,
                                   multiply_by=1)
    max_episodes = 100
    path_to_baseline_torch_model_state_dict = "/home/sascha/Documents/github.com/prl_baselines/data/ckpt.pt"
    agent_list = make_agents(env_wrapped, path_to_baseline_torch_model_state_dict)
    participants = make_participants(agent_list, starting_stack_size)
    experiment = PokerExperiment(
        env=env_wrapped,  # single environment to run sequential games on
        num_players=num_players,
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
        path_out='/prl/baselines/agents/eval/Pokersnowie2.txt',
        experiment=experiment,
        max_episodes_per_file=500)
