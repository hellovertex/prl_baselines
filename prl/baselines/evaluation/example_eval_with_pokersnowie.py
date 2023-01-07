from typing import List, Dict, Type, Tuple, Any

import gin
import gym
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env

from prl.baselines.agents.agents import CallingStation, StakePlayerImitator
from prl.baselines.agents.core.base_agent import RllibAgent
from prl.baselines.evaluation.core.experiment import PokerExperiment, PokerExperimentParticipant, \
    PokerExperiment_EarlyStopping
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


def make_participants(num_players: int,
                      agent_init_components: List[AGENT_INIT_COMPONENTS],
                      observation_space: gym.Space,
                      action_space: gym.Space) -> Tuple[PokerExperimentParticipant]:
    participants = []
    for i, (agent_cls, policy_config, stack) in enumerate(agent_init_components[:num_players]):
        agent_config = {'observation_space': observation_space,
                        'action_space': action_space,
                        'policy_config': policy_config}
        agent = agent_cls(agent_config)
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

# def get_heros(num_players):
#     if num_players == 2:
#         return [ "SB", "BB"]
#     if num_players:


if __name__ == '__main__':
    import gin

    gin.parse_config_file('../config.gin')

    starting_stack_size = 20000
    sb = 50
    bb = 100
    max_episodes = 2000
    num_players = 6
    # model_path = "/home/sascha/Documents/github.com/prl_baselines/data/ckpt(1).pt"
    model_path = get_prl_baseline_model_ckpt_path()
    baseline_v1 = (StakePlayerImitator,  # agent_cls
                   {'path_to_torch_model_state_dict': model_path},  # policy_config
                   starting_stack_size)
    calling_station = (CallingStation,
                       {},
                       starting_stack_size)
    agent_init_components = [
        baseline_v1,  # agent_cls, policy_config, stack
        baseline_v1,  # agent_cls, policy_config, stack
        baseline_v1,  # agent_cls, policy_config, stack
        baseline_v1,  # agent_cls, policy_config, stack
        baseline_v1,  # agent_cls, policy_config, stack
        baseline_v1  # agent_cls, policy_config, stack
    ]
    #for num_players in [2, 3, 4, 5, 6]:
    # for num_players in [2, 3, 4, 5, 6]:
    positions = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
    # for position in positions:  # [:num_players]
    env = init_wrapped_env(env_wrapper_cls=AugmentObservationWrapper,
                           stack_sizes=[starting_stack_size for _ in range(num_players)],
                           blinds=[sb, bb],
                           multiply_by=1)

    participants = make_participants(num_players,
                                     agent_init_components,
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
        from_action_plan=None,  # compute action from fixed series of actions instead of calls to agent.act
        # early_stopping_when=PokerExperiment_EarlyStopping.ALWAYS_REBUY_AND_PLAY_UNTIL_NUM_EPISODES_REACHED
    )
    # pos = position
    # if num_players == 2:
    #     pos = 'BTN' if position == 'BTN' else 'BB'
    db_gen = PokerExperimentToPokerSnowie().generate_database(
        path_out=str(get_snowie_database_output_path())+f'__n_players={num_players}__position={"ALL"}',
        experiment=experiment,
        max_episodes_per_file=1000,
        # hero_names=["StakePlayerImitator_Seat_1"]
        hero_names=positions
    )
"""
# todo: figure out how to ask ray.remote how many steps each actor can swing per seconds
        # todo run with 3+ players and see if it looks reasonable -- very important
        # todo in the baseline, if win_prob>.8 just make a bet here and there, we check way too often
        # [x] todo button is at 3 for >3 players, not >2, fix this
        # [x] todo player names
    
        # todo aws rl
        # todo very important fix the money in the experiment runnner
    
        # todo rake -- not so important
        # todo split pot -- we lose some games but only a very minot portion -- not important
    
        # todo how does it perform vs callingstation in terms of bb/100
        # todo initialize/bootstrap rl model with baseline NN
        # todo: what do we need from the run, metrics etc?
        # todo [optional] do we need to code up a purely MC based baseline?
        # todo: compute Poker Stats from baesline
        # todo: vpip 3bet etc
"""
