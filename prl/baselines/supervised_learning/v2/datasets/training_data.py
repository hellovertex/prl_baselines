import logging

import click

from prl.baselines.supervised_learning.v2.datasets.dataset_options import ActionGenOption, Stage, DatasetOptions
from prl.baselines.supervised_learning.v2.datasets.utils import parse_cmd_action_to_action_cls


@click.command()
@click.option("--num_top_players", default=20,
              type=int,
              help="How many top players hand histories should be used to generate the "
                   "data.")
@click.option("--nl",
              default='NL50',
              type=str,
              help="Which stakes the hand history belongs to."
                   "Determines the data directory.")
@click.option("--make_dataset_for_each_individual",
              default=False,
              type=bool,
              help="If True, creates a designated directory per player for "
                   "training data. Defaults to False.")
@click.option("--action_generation_option",
              default=ActionGenOption.no_folds_top_player_only_wins.value,
              type=int,
              help="Possible Values are \n"
                   "0: no_folds_top_player_all_showdowns\n"
                   "1: no_folds_top_player_only_wins\n"
                   "2: make_folds_from_top_players_with_randomized_hand\n"
                   "3: make_folds_from_showdown_loser_ignoring_rank\n"
                   "4: make_folds_from_fish\n"
                   "See `ActionGenOption`. ")
@click.option("--min_showdowns",
              default=5000,
              type=int,
              help="Minimum number of showdowns required to be eligible for top player "
                   "ranking. Default is 5 for debugging. 5000 is recommended for real "
                   "data.")
@click.option("--target_rounds",
              multiple=True,
              default=[  # Stage.PREFLOP.value,
                  Stage.FLOP.value,
                  Stage.TURN.value,
                  Stage.RIVER.value],
              type=int,
              help="Preprocessing will reduce data to the rounds specified. Possible values: "
                   "Stage.PREFLOP.value: 0\nStage.FLOP.value: 1\nStage.TURN.value: 2\nStage.RIVER.value: 3\n"
                   "Defaults to [FLOP,TURN,RIVER] rounds.")
@click.option("--action_space",
              default="ActionSpaceMinimal",
              type=str,
              help="Possible values are ActionSpace, ActionSpaceMinimal, FOLD, CHECK_CALL, RAISE")
def main(num_top_players,
         nl,
         make_dataset_for_each_individual,
         action_generation_option,
         min_showdowns,
         target_rounds,
         action_space):
    # Assumes raw_data.py has been ran to download and extract hand histories.
    opt = DatasetOptions(
        num_top_players=num_top_players,
        nl=nl,
        make_dataset_for_each_individual=make_dataset_for_each_individual,
        action_generation_option=ActionGenOption(action_generation_option),
        min_showdowns=min_showdowns,
        target_rounds=[Stage(x) for x in target_rounds],
        action_space=[parse_cmd_action_to_action_cls(action_space)]
    )


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()