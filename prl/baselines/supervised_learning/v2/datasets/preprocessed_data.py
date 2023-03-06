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

from prl.baselines.supervised_learning.v2.datasets.dataset_config import DatasetConfig, ActionGenOption, Stage
from prl.baselines.supervised_learning.v2.datasets.raw_data import make_raw_data_if_not_exists_already
from prl.baselines.supervised_learning.v2.datasets.top_player_selector import TopPlayerSelector
from prl.baselines.supervised_learning.v2.datasets.utils import parse_cmd_action_to_action_cls
from prl.baselines.supervised_learning.v2.datasets.vectorized_data import VectorizedData, \
    make_vectorized_data_if_not_exists_already
from prl.baselines.supervised_learning.v2.fast_hsmithy_parser import ParseHsmithyTextToPokerEpisode

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


class PreprocessedData:

    def __init__(self,
                 dataset_options: DatasetConfig,
                 parser_cls=ParseHsmithyTextToPokerEpisode):
        self.opt = dataset_options
        self.parser_cls = parser_cls
        dummy_env = init_wrapped_env(AugmentObservationWrapper,
                                     [5000 for _ in range(6)],
                                     blinds=(25, 50),
                                     multiply_by=1)
        self.env = dummy_env

    def maybe_encode_missing_data(self):

        selector = TopPlayerSelector(parser=self.parser_cls(self.opt))
        vectorized_data = VectorizedData(dataset_options=self.opt,
                                         parser_cls=self.parser_cls,
                                         top_player_selector=selector)
        vectorized_data.generate_missing(use_multiprocessing=False)

    def generate_missing(self):
        if os.path.exists(self.opt.dir_preprocessed_data):
            logging.info(f'Preprocessed data already exists at directory '
                         f'{self.opt.dir_preprocessed_data} '
                         f'for given configuration: {self.opt}')
        # else:
        self.maybe_encode_missing_data()
        # return
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
            # float to int if applicable
            df = df.apply(
                lambda x: x.apply(lambda y: np.int8(y) if int(y) == y else y))
            # int64 to int8 to save memory
            df = df.apply(pd.to_numeric, downcast='integer', errors='coerce').dropna()
            # todo: check why only preflop actions for
            #  data/02_vectorized/NL50/player_pool/no_folds_top_player_all_showdowns/Top20Players_n_showdowns=5000
            # maybe remove unused round
            for stage in list(Stage):
                if stage not in self.opt.target_rounds:
                    name = 'round_' + stage.name.casefold()
                    # todo: check if this actually removes the desired rows, consider adding axis=1?
                    df.drop(df[df[name] == 1], inplace=True)

            # maybe reduce action space
            if self.opt.action_space is ActionSpaceMinimal:
                assert df['label'].max < 3
            elif self.opt.action_space is ActionSpace:
                assert df['label'].min == min(ActionSpace).value
                assert df['label'].max == max(ActionSpace).value
            elif isinstance(self.opt.action_space, ActionSpaceMinimal):
                # todo: remove all labels that are greater than 2
                pass
            # write to disk
            filepath = os.path.join(self.opt.dir_preprocessed_data, Path(file).name + '.bz2')
            header = False
            if not os.path.exists(filepath):
                os.makedirs(os.path.realpath(Path(filepath).parent), exist_ok=True)
                df.columns = feature_names
                header = True
            df.to_csv(filepath,
                      index=True,
                      header=header,
                      index_label='label',
                      mode='a',
                      float_format='%.5f',
                      compression='bz2')


def make_preprocessed_data_if_not_exists_already(num_top_players,
                                                 nl,
                                                 from_gdrive_id,
                                                 make_dataset_for_each_individual,
                                                 action_generation_option,
                                                 use_multiprocessing,
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

    make_raw_data_if_not_exists_already(num_top_players, nl, from_gdrive_id)
    make_vectorized_data_if_not_exists_already(num_top_players,
                                               nl,
                                               from_gdrive_id,
                                               make_dataset_for_each_individual,
                                               action_generation_option,
                                               use_multiprocessing,
                                               min_showdowns)
    preprocessed_data = PreprocessedData(opt)
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
    make_preprocessed_data_if_not_exists_already(num_top_players,
                                                 nl,
                                                 from_gdrive_id,
                                                 make_dataset_for_each_individual,
                                                 action_generation_option,
                                                 use_multiprocessing,
                                                 min_showdowns,
                                                 target_rounds,
                                                 action_space)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
