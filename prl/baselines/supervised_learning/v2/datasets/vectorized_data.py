import glob
import logging
import os
from pathlib import Path
from typing import List, Type, Dict, re

import click
import numpy as np
import pandas as pd
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env
from tqdm import tqdm

from prl.baselines.supervised_learning.v2.datasets.dataset_options import DatasetOptions, \
    ActionGenOption
from prl.baselines.supervised_learning.v2.datasets.raw_data import TopPlayerSelector
from prl.baselines.supervised_learning.v2.fast_hsmithy_parser import \
    ParseHsmithyTextToPokerEpisode
from prl.baselines.supervised_learning.v2.datasets.tmp import EncoderV2
from prl.baselines.supervised_learning.v2.poker_model import PokerEpisodeV2


class VectorizedData:
    def __init__(self,
                 dataset_options: DatasetOptions,
                 parser_cls: Type[ParseHsmithyTextToPokerEpisode],
                 top_player_selector: TopPlayerSelector):
        self.opt = dataset_options
        self.parser_cls = parser_cls
        self.parser = parser_cls(nl=self.opt.nl)  # todo replace nl= with opt=
        self.top_player_selector = top_player_selector

    def _generate_per_selected_player(self, filenames, encoder, use_multiprocessing):
        # todo: implement
        raise NotImplementedError

    def alias_player_rank_to_ingame_name(self, filename: str):
        selected_players = self.top_player_selector.get_top_n_players(
            self.opt.num_top_players)
        selected_player_names = list(selected_players.keys())
        # map `01_raw/NL50/selected_players/PlayerRank0008` to int(8)
        filename = Path(filename).name
        rank: int = int(re.search(r'\d+', filename).group())
        # map rank int to key index of top players
        for i, name in enumerate(selected_player_names):
            if i == rank:
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
                print(e)
                continue
            if not observations:
                continue
            if training_data is None:
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

    def flush_data_to_disk(self,
                           training_data: np.ndarray,
                           labels: np.ndarray,
                           feature_names: List[str]):
        if training_data is not None:
            columns = None
            header = False
            # write to self.opt.dir_vectorized_data
            file_path = os.path.abspath(
                f'{out_dir}/{player_name}/data.csv.bz2')
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

            # one hot encode button
            one_hot_btn = pd.get_dummies(df['btn_idx'], prefix='btn_idx')
            df = pd.concat([df, one_hot_btn], axis=1)
            df.drop('btn_idx', axis=1, inplace=True)

            df.to_csv(file_path,
                      index=True,
                      header=header,
                      index_label='label',
                      mode='a',
                      float_format='%.5f',
                      compression='bz2'
                      )
            return "Success"
        return "Failure"

    def _generate_player_pool_data(self,
                                   files,
                                   encoder,
                                   use_multiprocessing):
        # In: 01_raw/NL50/selected_players/`Player Ranks zfilled`
        # Out: 02_vectorized/NL50/player_pool/folds_from_top_players/TopNPlayers/
        # consider creating parser for multiprocessing use
        # single files, pretty large
        for filename in files:
            selected_players = self.alias_player_rank_to_ingame_name(filename)
            episodesV2 = self.parser.parse_file(filename)
            training_data, labels = self.encode_episodes(episodesV2,
                                                         encoder,
                                                         selected_players)
            self.flush_data_to_disk(training_data, labels, encoder.feature_names)
            # todo: deprecate new_txt_to_vector_encoder and make tmp.EncoderV2
            #  encoder V2 the new one

    def generate(self,
                 env=None,
                 encoder_cls=EncoderV2):
        if env is None:
            env = init_wrapped_env(AugmentObservationWrapper,
                                   [5000 for _ in range(6)],
                                   blinds=(25, 50),
                                   multiply_by=1)
        encoder = encoder_cls(env)
        filenames = glob.glob(f'{self.opt.dir_raw_data_top_players}**/*.txt',
                              recursive=True)
        if self.opt.make_dataset_for_each_individual:
            return self._generate_per_selected_player(filenames,
                                                      encoder,
                                                      use_multiprocessing=True)
        else:
            return self._generate_player_pool_data(filenames,
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
def main(num_top_players, nl, from_gdrive_id):
    opt = DatasetOptions(num_top_players, nl)
    # raw_data = RawData(dataset_options, top_player_selector)
    # raw_data.generate(from_gdrive_id)
    parser_cls = ParseHsmithyTextToPokerEpisode
    # write top players
    selector = TopPlayerSelector(parser=parser_cls(nl))
    # sharks: Dict = selector.get_top_n_players(num_top_players)

    vectorized_data = VectorizedData(dataset_options=opt,
                                     parser_cls=parser_cls,
                                     top_player_selector=selector)
    # todo: 1. get top players, write to disk and ONLY THEN
    #  2. start multiprocessing


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
