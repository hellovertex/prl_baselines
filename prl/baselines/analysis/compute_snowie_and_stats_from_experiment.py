"""
"""
from pathlib import Path

from prl.baselines.agents.dummy_agents import RandomAgent
from prl.baselines.agents.tianshou_agents import BaselineAgent
from prl.baselines.analysis.core.utils import make_experiment
from prl.baselines.evaluation.pokersnowie.export import PokerExperimentToPokerSnowie


def run_experiment(experiment,
                   path_out,
                   pname,
                   criterion,
                   verbose=True,
                   max_episodes_per_file=1000):
    stats = experiment.options['stats']
    # path_out = f'./stats_div_6/{pname.split("_")[0]}/'
    PokerExperimentToPokerSnowie().generate_database(
        verbose=verbose,
        path_out=path_out + f'/{pname}',
        experiment=experiment,
        max_episodes_per_file=max_episodes_per_file,
        # hero_names=["StakePlayerImitator_Seat_1"]
        hero_names=[pname]
    )
    # for pstat in stats:
    #     if pstat.pname == pname:
    #         pstat.to_disk(fpath=f'{path_out}/{pname}.json')


def main(max_episodes,
         num_players,
         max_episodes_per_file,
         verbose):
    # hidden_dims = [256, 256]  # if '[256]' in pname else [512]
    # pname = '2NL_256x2'
    # 2.5NL ckpt_abs_fpath = "/home/hellovertex/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_all_players/with_folds_2NL_all_players/ckpt_dir_[256, 256]_1e-06/ckpt.pt"

    ckpt_abs_fpath = "/home/hellovertex/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_selected_players/with_folds/ckpt_dir/Sakhacop_[256]_1e-06/ckpt.pt"
    ckpt_abs_fpath = "/home/hellovertex/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_all_players/no_folds_selected_players/ckpt_dir_[512]_1e-06/ckpt.pt"
    ckpt_abs_fpath = "/home/hellovertex/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/from_all_players/all_games_and_folds_rand_cards_selected_players/ckpt_dir_[512]_1e-06/ckpt.pt"
    pname = Path(ckpt_abs_fpath).parent.stem
    pname = 'AI_AGENT_v2'
    hidden_dims = [256] if '256' in pname else [512]
    path_out = "./results_final"
    agent_names = [f'{pname}', '2', '3', '4', '5', '6']
    # make self play agents
    agents = [BaselineAgent(ckpt_abs_fpath,  # MajorityBaseline
                            flatten_input=False,
                            num_players=num_players,
                            model_hidden_dims=hidden_dims)]  # dont make self play agents
    agents += [RandomAgent() for _ in range(num_players - 1)]  # make random opponents instead

    # agent_names = [f'{pname}_{i}' for i in range(num_players)]
    experiment = make_experiment(max_episodes=max_episodes,
                                 num_players=num_players,
                                 agents=agents,
                                 agent_names=agent_names,
                                 ckpt_abs_fpath=ckpt_abs_fpath,
                                 hidden_dims=hidden_dims)
    # analyze all games by player
    run_experiment(experiment=experiment,
                   path_out=path_out,
                   pname=pname,
                   criterion=None,
                   verbose=verbose,
                   max_episodes_per_file=max_episodes_per_file)


if __name__ == '__main__':
    main(max_episodes=1000,
         num_players=6,
         max_episodes_per_file=1000,
         verbose=False)
