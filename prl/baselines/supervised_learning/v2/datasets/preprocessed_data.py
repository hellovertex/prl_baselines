import glob
import logging
import os
from pathlib import Path
from typing import Type, Union

import click
import numpy as np
import pandas as pd
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.base import ActionSpace, ActionSpaceMinimal
from prl.environment.Wrappers.utils import init_wrapped_env

from prl.baselines.supervised_learning.v2.datasets.dataset_config import DatasetConfig, \
    ActionGenOption, Stage
from prl.baselines.supervised_learning.v2.datasets.dataset_config import (
    arg_num_top_players,
    arg_nl,
    arg_from_gdrive_id,
    arg_make_dataset_for_each_individual,
    arg_action_generation_option,
    arg_use_multiprocessing,
    arg_min_showdowns,
    arg_target_rounds,
    arg_action_space
)
from prl.baselines.supervised_learning.v2.datasets.raw_data import \
    make_raw_data_if_not_exists_already
from prl.baselines.supervised_learning.v2.datasets.vectorized_data import \
    make_vectorized_data_if_not_exists_already
from prl.baselines.supervised_learning.v2.fast_hsmithy_parser import \
    ParseHsmithyTextToPokerEpisode


class PreprocessedData:

    def __init__(self,
                 dataset_config: DatasetConfig,
                 parser_cls=ParseHsmithyTextToPokerEpisode):
        self.opt = dataset_config
        self.parser_cls = parser_cls
        dummy_env = init_wrapped_env(AugmentObservationWrapper,
                                     [5000 for _ in range(6)],
                                     blinds=(25, 50),
                                     multiply_by=1)
        self.env = dummy_env

    @staticmethod
    def parse_cmd_action_to_action_cls(action: str) -> Union[
        ActionSpaceMinimal,  # Allow dichotomizers only for FOLD,CHECK,RAISE (Single bet size)
        Type[ActionSpaceMinimal],
        Type[ActionSpace]
    ]:
        if action == 'ActionSpace':
            return ActionSpace
        elif action == 'ActionSpaceMinimal':
            return ActionSpaceMinimal
        elif action.casefold().strip() == 'fold':
            return ActionSpaceMinimal.FOLD
        elif 'check' in action.casefold().strip():
            return ActionSpaceMinimal.CHECK_CALL
        elif action.casefold().strip() == 'raise':
            return ActionSpaceMinimal.RAISE
        else:
            raise NotImplementedError

    def generate_missing(self):
        if os.path.exists(self.opt.dir_preprocessed_data):
            logging.info(f'Preprocessed data already exists at directory '
                         f'{self.opt.dir_preprocessed_data} '
                         f'for given configuration: {self.opt}')
        # load .csv files into dataframe
        csv_files = glob.glob(self.opt.dir_vectorized_data + '/**/*.csv.bz2',
                              recursive=True)
        feature_names = list(self.env.obs_idx_dict.keys())
        for file in csv_files:
            df = pd.read_csv(file,
                             sep=',',
                             dtype='float32',
                             encoding='cp1252',
                             compression='bz2')
            if file == '/home/sascha/Documents/github.com/prl_baselines/prl/baselines' \
                       '/supervised_learning/v2/tests/data/02_vectorized/NL50' \
                       '/player_pool/folds_from_top_players_with_randomized_hand/Top20Players_n_showdowns=10/data_PlayerRank009355.csv.bz2':
                a = 1
                print('debug')
            # float to int if applicable
            df = df.apply(
                lambda x: x.apply(lambda y: np.int8(y) if int(y) == y else y))
            # int64 to int8 to save memory
            df = df.apply(pd.to_numeric, downcast='integer', errors='coerce').dropna()
            # maybe remove unused round
            for stage in list(Stage):
                if stage not in self.opt.target_rounds:
                    name = 'round_' + stage.name.casefold()
                    df = df[df[name] == 0]

            # maybe reduce action space
            if self.opt.action_space[0] is ActionSpaceMinimal:
                df[df['label'] > max(ActionSpaceMinimal)] = max(ActionSpaceMinimal)
            elif self.opt.action_space[0] is ActionSpace:
                assert df['label'].min == min(ActionSpace).value
                assert df['label'].max == max(ActionSpace).value
            elif isinstance(self.opt.action_space[0], ActionSpaceMinimal):
                target_action = self.opt.action_space[0].value
                df = df[df['label'] == target_action]
            # write to disk
            filepath = os.path.join(self.opt.dir_preprocessed_data,
                                    Path(file).stem + '.bz2')
            header = False
            # df.columns = feature_names
            if not os.path.exists(filepath):
                os.makedirs(os.path.realpath(Path(filepath).parent), exist_ok=True)
                header = True
            df.to_csv(filepath,
                      index=False,
                      header=header,
                      mode='a',
                      float_format='%.5f',
                      compression='bz2')


def make_preprocessed_data_if_not_exists_already(dataset_config,
                                                 use_multiprocessing):
    # Assumes raw_data.py has been ran to download and extract hand histories.
    make_raw_data_if_not_exists_already(dataset_config)
    make_vectorized_data_if_not_exists_already(dataset_config, use_multiprocessing)
    preprocessed_data = PreprocessedData(dataset_config)
    preprocessed_data.generate_missing()


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
def main(num_top_players,
         nl,
         from_gdrive_id,
         make_dataset_for_each_individual,
         action_generation_option,
         use_multiprocessing,
         min_showdowns,
         target_rounds,
         action_space):
    dataset_config = DatasetConfig(
        num_top_players=num_top_players,
        nl=nl,
        make_dataset_for_each_individual=make_dataset_for_each_individual,
        action_generation_option=ActionGenOption(action_generation_option),
        min_showdowns=min_showdowns,
        target_rounds=[Stage(x) for x in target_rounds],
        action_space=[PreprocessedData.parse_cmd_action_to_action_cls(action_space)]
    )
    make_preprocessed_data_if_not_exists_already(dataset_config, use_multiprocessing)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
