from typing import List, Dict

from prl.baselines.agents.agents import BaselineAgent
from prl.baselines.agents.eval.core.experiment import PokerExperimentParticipant, AGENT
from prl.baselines.agents.policies import CallingStation, StakeLevelImitationPolicy


def make_agents(env, path_to_torch_model_state_dict):
    policy_config = {'path_to_torch_model_state_dict': path_to_torch_model_state_dict}
    baseline_policy = StakeLevelImitationPolicy(env.observation_space, env.action_space, policy_config)
    reference_policy = CallingStation(env.observation_space, env.action_space, policy_config)

    baseline_agent = BaselineAgent({'rllib_policy': baseline_policy})
    reference_agent = BaselineAgent({'rllib_policy': reference_policy})
    return [baseline_agent, reference_agent]


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
