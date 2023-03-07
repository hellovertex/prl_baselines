from __future__ import annotations

import logging

import click
from hydra import compose, initialize
from omegaconf import DictConfig

from prl.baselines.supervised_learning.v2.action_prediction.nn_training.train_eval import \
    train_eval
from prl.baselines.supervised_learning.v2.action_prediction.nn_training.training_config import \
    TrainingParams
from prl.baselines.supervised_learning.v2.datasets.dataset_config import (
    ActionGenOption,
    Stage,
    DatasetConfig,
    arg_num_top_players,
    arg_nl,
    arg_from_gdrive_id,
    arg_make_dataset_for_each_individual,
    arg_action_generation_option,
    arg_use_multiprocessing,
    arg_min_showdowns,
    arg_target_rounds,
    arg_action_space,
    arg_sub_sampling_technique,
    arg_seed_dataset
)
from prl.baselines.supervised_learning.v2.datasets.training_data import get_datasets
from prl.baselines.supervised_learning.v2.datasets.utils import \
    parse_cmd_action_to_action_cls


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
@arg_seed_dataset
@click.option('--train_configfile',
              type=str,
              default='config.yaml',
              help="Name of .yaml file inside ./conf .")
def main(num_top_players,
         nl,
         from_gdrive_id,
         make_dataset_for_each_individual,
         action_generation_option,
         use_multiprocessing,
         min_showdowns,
         target_rounds,
         action_space,
         sub_sampling_technique,
         seed,
         train_configfile):
    dataset_config = DatasetConfig(
        num_top_players=num_top_players,
        nl=nl,
        from_gdrive_id=from_gdrive_id,
        make_dataset_for_each_individual=make_dataset_for_each_individual,
        action_generation_option=ActionGenOption(action_generation_option),
        min_showdowns=min_showdowns,
        target_rounds=[Stage(x) for x in target_rounds],
        action_space=[parse_cmd_action_to_action_cls(action_space)],
        sub_sampling_technique=sub_sampling_technique,
        seed=seed
    )
    # 1. Make dataset, from scratch if necessary
    # (downloading, extracting, encoding, vectorizing, preprocessing)
    train, test, label_weights = get_datasets(dataset_config, use_multiprocessing)

    # 2. Load training params from .yaml config
    initialize(version_base=None, config_path="conf")
    cfg: DictConfig = compose(train_configfile)
    params = TrainingParams(**cfg)

    # 3. Run Training
    train_eval(params)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
