from __future__ import annotations

import glob
import logging
import multiprocessing
import time
from functools import partial

import click
from hydra import compose, initialize
from omegaconf import DictConfig

from prl.baselines.supervised_learning.v2.action_prediction.nn_training.train_eval import \
    TrainEval, train_eval
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
    arg_min_showdowns,
    arg_target_rounds,
    arg_action_space,
    arg_sub_sampling_technique,
    arg_seed_dataset, arg_hudstats
)
from prl.baselines.supervised_learning.v2.datasets.utils import \
    parse_cmd_action_to_action_cls


@click.command()
@arg_num_top_players
@arg_nl
@arg_from_gdrive_id
@arg_make_dataset_for_each_individual
@arg_action_generation_option
@arg_min_showdowns
@arg_target_rounds
@arg_action_space
@arg_sub_sampling_technique
@arg_seed_dataset
@arg_hudstats
@click.option('--train_configfile',
              type=str,
              default='config.yaml',
              help="Name of .yaml file inside ./conf .")
def main(num_top_players,
         nl,
         from_gdrive_id,
         make_dataset_for_each_individual,
         action_generation_option,
         min_showdowns,
         hudstats,
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
        hudstats_enabled=hudstats,
        target_rounds=[Stage(x) for x in target_rounds],
        action_space=[parse_cmd_action_to_action_cls(action_space)],
        sub_sampling_technique=sub_sampling_technique,
        seed=seed,
    )

    # Load training params from .yaml config
    initialize(version_base=None, config_path="conf")
    cfg: DictConfig = compose(train_configfile)
    params = TrainingParams(**cfg)

    # Run Training
    data_files = glob.glob(dataset_config.dir_preprocessed_data +
                           '/**/*.csv.bz2',
                           recursive=True)
    if dataset_config.make_dataset_for_each_individual:
        train_eval_fn = partial(train_eval, params=params, dataset_config=dataset_config)
        start = time.time()
        p = multiprocessing.Pool()
        t0 = time.time()
        for x in p.imap_unordered(train_eval_fn, data_files):
            print(x + f'. Took {time.time() - t0} seconds')
        print(f'Finished job after {time.time() - start} seconds.')
        p.close()
    else:
        TrainEval(dataset_config).run(params, data_files)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
