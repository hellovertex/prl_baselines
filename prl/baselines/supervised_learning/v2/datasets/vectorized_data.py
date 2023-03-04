import glob
import logging
import os
from pathlib import Path
from typing import List, Type, Dict, Optional
import re

import click
import numpy as np
import pandas as pd
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env
from tqdm import tqdm

from prl.baselines.supervised_learning.v2.datasets.dataset_options import DatasetOptions, \
    ActionGenOption
from prl.baselines.supervised_learning.v2.datasets.raw_data import TopPlayerSelector, \
    RawData
from prl.baselines.supervised_learning.v2.fast_hsmithy_parser import \
    ParseHsmithyTextToPokerEpisode
from prl.baselines.supervised_learning.v2.datasets.tmp import EncoderV2
from prl.baselines.supervised_learning.v2.poker_model import PokerEpisodeV2


class PersistantStorage:
    def __init__(self, dataset_options):
        self.opt = dataset_options
        self.num_files_written_to_disk = 0

    def flush_data_to_disk(self,
                           training_data: np.ndarray,
                           labels: np.ndarray,
                           feature_names: List[str],
                           compression='.bz2'  # set to '' if you want to save raw .csv
                           ):
        if training_data is not None:
            columns = None
            header = False
            # write to self.opt.dir_vectorized_data
            file_path = os.path.join(self.opt.dir_vectorized_data,
                                     f'data_'
                                     f'{str(self.num_files_written_to_disk).zfill(3)}'
                                     f'.csv{compression}')
            if not os.path.exists(Path(file_path).parent):
                os.makedirs(os.path.realpath(Path(file_path).parent), exist_ok=True)
            if not os.path.exists(file_path):
                columns = feature_names
                header = True
            df = pd.DataFrame(data=training_data,
                              index=labels,  # The index (row labels) of the DataFrame.
                              columns=columns)
            # float to int if applicable
            df = df.apply(lambda x: x.apply(lambda y: np.int8(y) if int(y) == y else y))

            # # one hot encode button -- done by CanonicalVectorizer now
            # one_hot_btn = pd.get_dummies(df['btn_idx'], prefix='btn_idx')
            # df = pd.concat([df, one_hot_btn], axis=1)
            # df.drop('btn_idx', axis=1, inplace=True)

            df.to_csv(file_path,
                      index=True,
                      header=header,
                      index_label='label',  # index=False ?
                      mode='a',
                      float_format='%.5f',
                      compression='bz2'
                      )
            return "Success"
        return "Failure"


class VectorizedData:
    def __init__(self,
                 dataset_options: DatasetOptions,
                 parser_cls: Type[ParseHsmithyTextToPokerEpisode],
                 top_player_selector: TopPlayerSelector,
                 storage: Optional[PersistantStorage] = None):
        self.opt = dataset_options
        self.parser_cls = parser_cls
        self.parser = parser_cls(nl=self.opt.nl)  # todo replace nl= with opt=
        self.top_player_selector = top_player_selector
        self.storage = storage if storage else PersistantStorage(self.opt)
        self.num_files_written_to_disk = 0

    def alias_player_rank_to_ingame_name(self, selected_player_names, filename: str):
        # map `01_raw/NL50/selected_players/PlayerRank0008` to int(8)
        filename = Path(filename).name
        rank: int = int(re.search(r'\d+', filename).group())
        # map rank int to key index of top players
        for i, name in enumerate(selected_player_names):
            if i + 1 == rank:
                return [name]
        raise ValueError(f"No Player name has been found for file {filename}")

    def encode_episodes(self,
                        episodesV2: List[PokerEpisodeV2],
                        encoder: EncoderV2,
                        selected_players: List[str]):
        training_data, labels = None, None
        for ep in tqdm(episodesV2):
            try:
                observations, actions = encoder.encode_episode(
                    ep,
                    a_opt=self.opt.action_generation_option,
                    limit_num_players=5,
                    selected_players=selected_players,
                    verbose=False)
            except Exception as e:
                raise e
                # print(e)
                # continue
            if not observations:
                continue
            elif training_data is None:
                training_data = observations
                labels = actions
            else:
                try:
                    training_data = np.concatenate((training_data, observations),
                                                   axis=0)
                    labels = np.concatenate((labels, actions), axis=0)
                except Exception as e:
                    print(e)
        return training_data, labels

    def _generate_per_selected_player(self,
                                      selected_player_names: List[str],
                                      files: List[str],
                                      encoder: EncoderV2,
                                      use_multiprocessing: bool):
        # todo: implement
        raise NotImplementedError

    def _generate_player_pool_data(self,
                                   selected_player_names: List[str],
                                   files: List[str],
                                   encoder: EncoderV2,
                                   use_multiprocessing: bool):
        # In: 01_raw/NL50/selected_players/`Player Ranks zfilled`
        # Out: 02_vectorized/NL50/player_pool/folds_from_top_players/TopNPlayers/
        # consider creating parser for multiprocessing use
        # single files, pretty large
        for filename in files:
            # filename to playername to one selected_players
            selected_players = self.alias_player_rank_to_ingame_name(
                selected_player_names, filename)
            episodesV2 = self.parser.parse_file(filename)
            training_data, labels = self.encode_episodes(episodesV2,
                                                         encoder,
                                                         selected_players)
            self.storage.flush_data_to_disk(training_data, labels,
                                            encoder.feature_names)
            # todo: deprecate new_txt_to_vector_encoder and make tmp.EncoderV2
            #  encoder V2 the new one

    def _make_missing_data(self):
        # extracts hand histories for Top M players,
        # where M=self.opt.num_top_players-missing
        RawData(self.opt, self.top_player_selector).generate()

    def generate(self,
                 env=None,
                 encoder_cls=EncoderV2):
        if env is None:
            env = init_wrapped_env(AugmentObservationWrapper,
                                   [5000 for _ in range(6)],
                                   blinds=(25, 50),
                                   multiply_by=1)
        encoder = encoder_cls(env)

        if not self.opt.exists_raw_data_for_all_selected_players():
            self._make_missing_data()

        filenames = glob.glob(f'{self.opt.dir_raw_data_top_players}/**/*.txt',
                              recursive=True)

        selected_players = self.top_player_selector.get_top_n_players_min_showdowns(
            self.opt.num_top_players, self.opt.min_showdowns)

        if self.opt.make_dataset_for_each_individual:
            return self._generate_per_selected_player(list(selected_players.keys()),
                                                      filenames,
                                                      encoder,
                                                      use_multiprocessing=True)
        else:
            return self._generate_player_pool_data(list(selected_players.keys()),
                                                   filenames,
                                                   encoder,
                                                   use_multiprocessing=True)


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
              default=False,
              type=int,
              help="Possible Values are \n"
                   "0: no_folds_top_player_all_showdowns\n"
                   "1: no_folds_top_player_only_wins\n"
                   "2: make_folds_from_top_players_with_randomized_hand\n"
                   "3: make_folds_from_showdown_loser_ignoring_rank\n"
                   "4: make_folds_from_fish\n"
                   "See `ActionGenOption`. ")
def main(num_top_players,
         nl,
         make_dataset_for_each_individual,
         action_generation_option):
    # Assumes raw_data.py has been ran to download and extract hand histories.
    opt = DatasetOptions(
        num_top_players=num_top_players,
        nl=nl,
        make_dataset_for_each_individual=make_dataset_for_each_individual,
        action_generation_option=ActionGenOption(action_generation_option),
        min_showdowns=5
    )
    parser_cls = ParseHsmithyTextToPokerEpisode
    # write top players
    selector = TopPlayerSelector(parser=parser_cls(nl))
    vectorized_data = VectorizedData(dataset_options=opt,
                                     parser_cls=parser_cls,
                                     top_player_selector=selector)
    vectorized_data.generate()
    # todo: test
    # todo: multiprocessing


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
