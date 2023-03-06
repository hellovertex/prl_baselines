import logging

import click

from prl.baselines.supervised_learning.v2.datasets.dataset_config import ActionGenOption, Stage, DatasetConfig
from prl.baselines.supervised_learning.v2.datasets.utils import parse_cmd_action_to_action_cls
from prl.baselines.supervised_learning.v2.datasets.dataset_config import (
    arg_num_top_players,
    arg_nl,
    arg_from_gdrive_id,
    arg_make_dataset_for_each_individual,
    arg_action_generation_option,
    arg_use_multiprocessing,
    arg_min_showdowns,
    arg_target_rounds,
    arg_action_space,
    arg_sub_sampling_technique
)

@click.command()
@arg_num_top_players
@arg_nl
@arg_from_gdrive_id
@arg_make_dataset_for_each_individual
@arg_action_generation_option
@arg_use_multiprocessing
@arg_min_showdowns
@arg_target_rounds
@arg_action_space
@arg_sub_sampling_technique
def main(num_top_players,
         nl,
         make_dataset_for_each_individual,
         action_generation_option,
         min_showdowns,
         target_rounds,
         action_space):
    # Assumes raw_data.py has been ran to download and extract hand histories.
    opt = DatasetConfig(
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