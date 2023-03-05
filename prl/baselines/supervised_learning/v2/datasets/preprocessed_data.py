import glob
import logging
import os
from typing import Type, Union

import click
import numpy as np
import pandas as pd
from prl.environment.Wrappers.base import ActionSpace, ActionSpaceMinimal

from prl.baselines.supervised_learning.v2.datasets.dataset_options import DatasetOptions, ActionGenOption, Stage
from prl.baselines.supervised_learning.v2.datasets.top_player_selector import TopPlayerSelector
from prl.baselines.supervised_learning.v2.datasets.vectorized_data import VectorizedData
from prl.baselines.supervised_learning.v2.fast_hsmithy_parser import ParseHsmithyTextToPokerEpisode


class PreprocessedData:

    def __init__(self,
                 dataset_options: DatasetOptions):
        self.opt = dataset_options

    def maybe_encode_missing_data(self):
        parser_cls = ParseHsmithyTextToPokerEpisode
        selector = TopPlayerSelector(parser=parser_cls(self.opt.nl))
        vectorized_data = VectorizedData(dataset_options=self.opt,
                                         parser_cls=parser_cls,
                                         top_player_selector=selector)
        vectorized_data.generate(use_multiprocessing=False)

    def generate(self):
        if os.path.exists(self.opt.dir_preprocessed_data):
            logging.info(f'Preprocessed data already exists at directory '
                         f'{self.opt.dir_preprocessed_data} '
                         f'for given configuration: {self.opt}')
        else:
            self.maybe_encode_missing_data()
            # return
            # load .csv files into dataframe
            csv_files = glob.glob(self.opt.dir_vectorized_data + '/**/*.csv.bz2',
                                  recursive=True)
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
                # maybe reduce action space
                # write to disk


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
    preprocessed_data = PreprocessedData(opt)
    preprocessed_data.generate()
    # top_player_selector = TopPlayerSelector(parser)
    # raw_data = RawData(dataset_options, top_player_selector)
    # raw_data.generate(from_gdrive_id)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
