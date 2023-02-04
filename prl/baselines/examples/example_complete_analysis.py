import os
from pathlib import Path
from typing import Dict, Type, Tuple, Any

import click
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env

from prl.baselines.agents.core.base_agent import RllibAgent
from prl.baselines.agents.tianshou_agents import BaselineAgent
from prl.baselines.agents.tianshou_policies import default_rainbow_params, get_rainbow_config
from prl.baselines.evaluation.core.experiment import PokerExperiment, PokerExperimentParticipant, make_participants
from prl.baselines.evaluation.pokersnowie.export import PokerExperimentToPokerSnowie
from prl.baselines.evaluation.utils import get_default_env
from prl.baselines.examples.examples_tianshou_env import MCAgent, make_default_tianshou_env

def run_analysis_from_position():
    pass

def run_analysis_single_player(pname, ckpt_abs_fpath):
    num_players = 6
    positions_two = ["BTN", "BB"]
    positions_multi = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
    positions = positions_two if num_players == 2 else positions_multi[:num_players]

    max_episodes = 100
    verbose = True
    hidden_dims = [256]
    starting_stack = 20000
    stack_sizes = [starting_stack for _ in range(num_players)]
    agent_names = [f'{pname}_{i}' for i in range(num_players)]
    env = make_default_tianshou_env(mc_model_ckpt_path=None,  # dont use mc
                                    agents=agent_names,
                                    num_players=len(agent_names))

    # make self play agents
    agents = [BaselineAgent(ckpt_abs_fpath,
                          flatten_input=False,
                          num_players=num_players,
                          model_hidden_dims=hidden_dims) for _ in range(num_players)]
    assert len(agents) == num_players == len(stack_sizes)
    participants = make_participants(agents, starting_stack)

    # run self play
    experiment = PokerExperiment(
        # env
        wrapped_env=env,  # single environment to run sequential games on
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

    db_gen = PokerExperimentToPokerSnowie().generate_database(
        verbose=verbose,
        path_out=f'./pokersnowie/ilaviiitech_256',
        experiment=experiment,
        max_episodes_per_file=1000,
        # hero_names=["StakePlayerImitator_Seat_1"]
        hero_names=positions
    )

def main(input_folder):
    """
    input_folder subdirs per player. script computes
    i) pokersnowie exports
     - per player
     -- per position BTN, SB, BB, UTG, MP, CO
     - per pool
     -- per position BTN, SB, BB, UTG, MP, CO
     # todo consider adding this for NN part: with pseudo-harmonic action mapping enabled and disabled
    ii) stats analysis given showdown
     - per player
     - per pool
    iii) with pseudo-harmonic action mapping enabled and disabled

    out:
    directory containing subfolder per player and one for pool,
    as well as two json files containing the final player stats.
    """
    # Input: Playername or Pool
    # Position
    # Harmonic Mapping
    # Output: Corresponding SnowieDatabase and Stat analysis
    player_dirs = [x[0] for x in
                   os.walk("/home/sascha/Documents/github.com/prl_baselines/data/new_snowie/with_folds/ckpt_dir")][1:]

    for pdir in player_dirs:
        if not Path(pdir).stem == 'ckpt':
            run_analysis_single_player(pname=Path(pdir).stem,
                                       ckpt_abs_fpath=pdir+'/ckpt.pt')



if __name__ == '__main__':
    input_folder = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data_test"
    main(input_folder)
