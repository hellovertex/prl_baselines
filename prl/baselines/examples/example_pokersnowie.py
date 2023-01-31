from typing import List, Dict, Type, Tuple, Any

import gin
import gym
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env

from prl.baselines.agents.core.base_agent import RllibAgent
from prl.baselines.evaluation.core.experiment import PokerExperiment, PokerExperimentParticipant, \
    PokerExperiment_EarlyStopping
from prl.baselines.evaluation.pokersnowie.export import PokerExperimentToPokerSnowie

AGENT_CLS = Type[RllibAgent]
POLICY_CONFIG = Dict[str, Any]
STARTING_STACK = int
AGENT_INIT_COMPONENTS = Tuple[AGENT_CLS, POLICY_CONFIG, STARTING_STACK]


def make_participants(observation_space: gym.Space,
                      action_space: gym.Space) -> Tuple[PokerExperimentParticipant]:

    participants = []
    participants.append(PokerExperimentParticipant(id=i,
                                                   name=f'{agent_cls.__name__}_Seat_{i + 1}',
                                                   alias=f'Agent_{i}',
                                                   starting_stack=stack,
                                                   agent=agent,
                                                   config={}))
    return tuple(participants)


@gin.configurable
def get_prl_baseline_model_ckpt_path(path=""):
    """Passes path from config.gin file to the caller """
    return path


@gin.configurable
def get_snowie_database_output_path(path=""):
    """Passes path from config.gin file to the caller """
    return path


if __name__ == '__main__':
    max_episodes = 100
    num_players = 6
    starting_stacks = [20000 for _ in range(num_players)]
    positions = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
    # for position in positions:  # [:num_players]
    env = init_wrapped_env(env_wrapper_cls=AugmentObservationWrapper,
                           stack_sizes=starting_stacks,
                           blinds=[50, 100],
                           multiply_by=1)
    agents = []
    participants = make_participants(
                                    observation_space=env.observation_space,
                                     action_space=env.action_space)
    experiment = PokerExperiment(
        # env
        env=env,  # single environment to run sequential games on
        num_players=num_players,
        starting_stack_sizes=starting_stacks,
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
    # pos = position
    # if num_players == 2:
    #     pos = 'BTN' if position == 'BTN' else 'BB'
    db_gen = PokerExperimentToPokerSnowie().generate_database(
        path_out=str(get_snowie_database_output_path()) + f'__n_players={num_players}__position={"000000"}',
        experiment=experiment,
        max_episodes_per_file=1000,
        # hero_names=["StakePlayerImitator_Seat_1"]
        hero_names=positions
    )
