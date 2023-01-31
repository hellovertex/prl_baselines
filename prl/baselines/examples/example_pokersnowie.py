from typing import List, Dict, Type, Tuple, Any

import gin
import gym
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env

from prl.baselines.agents.core.base_agent import RllibAgent
from prl.baselines.agents.tianshou_policies import default_rainbow_params, get_rainbow_config, MCPolicy
from prl.baselines.evaluation.core.experiment import PokerExperiment, PokerExperimentParticipant, \
    PokerExperiment_EarlyStopping
from prl.baselines.evaluation.pokersnowie.export import PokerExperimentToPokerSnowie
from prl.baselines.examples.examples_tianshou_env import make_vector_env, MCAgent

AGENT_CLS = Type[RllibAgent]
POLICY_CONFIG = Dict[str, Any]
STARTING_STACK = int
AGENT_INIT_COMPONENTS = Tuple[AGENT_CLS, POLICY_CONFIG, STARTING_STACK]


def make_participants(agents, starting_stack) -> Tuple[PokerExperimentParticipant]:
    participants = []
    for i, agent in enumerate(agents):
        participants.append(PokerExperimentParticipant(id=i,
                                                       name=f'{type(agent).__name__}_Seat_{i + 1}',
                                                       alias=f'Agent_{i}',
                                                       starting_stack=starting_stack,
                                                       agent=agent,
                                                       config={}))
    return tuple(participants)


if __name__ == '__main__':
    max_episodes = 100
    # environment config
    num_players = 2
    starting_stack = 20000
    stack_sizes = [starting_stack for _ in range(num_players)]
    agent_names = [f'p{i}' for i in range(num_players)]
    rainbow_config = get_rainbow_config(default_rainbow_params)

    env_config = {"env_wrapper_cls": AugmentObservationWrapper,
                  # "stack_sizes": [100, 125, 150, 175, 200, 250],
                  "stack_sizes": stack_sizes,
                  "multiply_by": 1,  # use 100 for floats to remove decimals but we have int stacks
                  "scale_rewards": False,  # we do this ourselves
                  "blinds": [50, 100]}
    # env = init_wrapped_env(**env_config)
    # obs0 = env.reset(config=None)
    num_envs = 31
    ckpt = ""
    env = init_wrapped_env(**env_config)

    agents = [
        MCAgent(ckpt),
        MCAgent(ckpt),
        MCAgent(ckpt),
    ]

    participants = make_participants(agents, starting_stack)
    experiment = PokerExperiment(
        # env
        env=env,  # single environment to run sequential games on
        num_players=num_players,
        starting_stack_sizes=stack_sizes,
        env_reset_config=None,  # can pass {'deck_state_dict': Dict[str, Any]} to init the deck and player cards
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
    positions_two = ["BTN", "BB"]
    positions_multi = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
    positions = positions_two if num_players == 2 else positions_multi[:num_players]
    db_gen = PokerExperimentToPokerSnowie().generate_database(
        path_out='./pokersnowie',
        experiment=experiment,
        max_episodes_per_file=1000,
        # hero_names=["StakePlayerImitator_Seat_1"]
        hero_names=positions
    )
