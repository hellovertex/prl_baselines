import os
from pathlib import Path

from prl.baselines.agents.tianshou_agents import BaselineAgent, MajorityBaseline
from prl.baselines.evaluation.core.experiment import PokerExperiment, make_participants
from prl.baselines.evaluation.pokersnowie.export import PokerExperimentToPokerSnowie
from prl.baselines.evaluation.stats import PlayerStats
from prl.baselines.examples.examples_tianshou_env import make_default_tianshou_env


def run_analysis_from_position():
    pass


def run_experiment(experiment,
                   pname,
                   criterion,
                   verbose=True,
                   max_episodes_per_file=1000):
    stats = experiment.options['stats']
    path_out = f'./pokersnowie/{pname.split("_")[0]}/'
    PokerExperimentToPokerSnowie().generate_database(
        verbose=verbose,
        path_out=path_out + f'{pname}_{criterion}',
        experiment=experiment,
        max_episodes_per_file=max_episodes_per_file,
        # hero_names=["StakePlayerImitator_Seat_1"]
        hero_names=[criterion]
    )
    for pstat in stats:
        pstat.to_disk(fpath=f'{path_out}/{pname}.json')


def run_analysis_majority_baseline(max_episodes, ckpts):
    pname = "MajorityVoting"
    num_players = 6
    positions_two = ["BTN", "BB"]
    positions_multi = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
    positions = positions_two if num_players == 2 else positions_multi[:num_players]

    max_episodes_per_file = 1000
    verbose = True
    hidden_dims = [[256] if '[256]' in pname else [512] for pname in ckpts]
    starting_stack = 20000
    stack_sizes = [starting_stack for _ in range(num_players)]
    agent_names = [f'{pname}_{i}' for i in range(num_players)]
    env = make_default_tianshou_env(mc_model_ckpt_path=None,  # dont use mc
                                    agents=agent_names,
                                    num_players=len(agent_names))

    agents = [MajorityBaseline(ckpts,
                               model_hidden_dims=hidden_dims,
                               flatten_input=False,
                               num_players=num_players) for _ in range(num_players)]

    assert len(agents) == num_players == len(stack_sizes)
    participants = make_participants(agents, starting_stack)
    stats = [PlayerStats(pname=pname) for pname in agent_names]
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
        options={'stats': stats},
        # actors
        participants=participants,  # wrapper around agents that hold rllib policies that act given observation
        from_action_plan=None,  # compute action from fixed series of actions instead of calls to agent.act
        # early_stopping_when=PokerExperiment_EarlyStopping.ALWAYS_REBUY_AND_PLAY_UNTIL_NUM_EPISODES_REACHED
    )
    # analyze all games by player
    run_experiment(experiment=experiment,
                   pname=pname,
                   criterion=None,
                   verbose=verbose,
                   max_episodes_per_file=max_episodes_per_file)
    # because we can not use every game if we only look for a specific table position
    experiment.max_episodes *= num_players
    for pos in positions:
        # we have to call with heronames=[pos] instead of heronames=[positions]
        # because pokersnowie will only look at one playername per episode
        # but heronames=[positions] would generate all positions/playernames in one episode
        # so we have to run `max_episodes_per_file` episode per position instead
        # analyze games where player was sitting at position pos
        run_experiment(experiment=experiment,
                       pname=pname,
                       criterion=pos,
                       verbose=verbose,
                       max_episodes_per_file=max_episodes_per_file)


def run_analysis_single_baseline(max_episodes, pname, ckpt_abs_fpath):
    num_players = 6
    positions_two = ["BTN", "BB"]
    positions_multi = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
    positions = positions_two if num_players == 2 else positions_multi[:num_players]

    max_episodes_per_file = 1000
    verbose = True
    hidden_dims = [256] if '[256]' in pname else [512]
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
                            model_hidden_dims=hidden_dims) for _ in range(num_players)]  # make self play agents

    assert len(agents) == num_players == len(stack_sizes)
    participants = make_participants(agents, starting_stack)
    stats = [PlayerStats(pname=pname) for pname in agent_names]
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
        options={'stats': stats},
        # actors
        participants=participants,  # wrapper around agents that hold rllib policies that act given observation
        from_action_plan=None,  # compute action from fixed series of actions instead of calls to agent.act
        # early_stopping_when=PokerExperiment_EarlyStopping.ALWAYS_REBUY_AND_PLAY_UNTIL_NUM_EPISODES_REACHED
    )
    # analyze all games by player
    run_experiment(experiment=experiment,
                   pname=pname,
                   criterion=None,
                   verbose=verbose,
                   max_episodes_per_file=max_episodes_per_file)
    # because we can not use every game if we only look for a specific table position
    # experiment.max_episodes *= num_players
    # for pos in positions:
    #     # we have to call with heronames=[pos] instead of heronames=[positions]
    #     # because pokersnowie will only look at one playername per episode
    #     # but heronames=[positions] would generate all positions/playernames in one episode
    #     # so we have to run `max_episodes_per_file` episode per position instead
    #     # analyze games where player was sitting at position pos
    #     run_experiment(experiment=experiment,
    #                    pname=pname,
    #                    criterion=pos,
    #                    verbose=verbose,
    #                    max_episodes_per_file=max_episodes_per_file)


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
    input_folder = "/home/hellovertex/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_selected_players/with_folds/ckpt_dir"
    # Input: Playername or Pool
    # Position
    # Harmonic Mapping
    # Output: Corresponding SnowieDatabase and Stat analysis
    player_dirs = [x[0] for x in
                   os.walk(input_folder)][1:]
    player_dirs = [pdir for pdir in player_dirs if not Path(pdir).stem == 'ckpt']
    # for pdir in player_dirs:
    #     if not Path(pdir).stem == 'ckpt':
    #         # baseline analysis goes by position
    #         run_analysis_single_baseline(max_episodes=100,
    #                                      pname=Path(pdir).stem,
    #                                      ckpt_abs_fpath=pdir + '/ckpt.pt')
    #         # selected_player analysis goes by available .txt data
    ckpts = [pdir + '/ckpt.pt' for pdir in player_dirs]
    run_analysis_majority_baseline(max_episodes=10000, ckpts=ckpts)


if __name__ == '__main__':
    input_folder = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data_test"
    main(input_folder)
