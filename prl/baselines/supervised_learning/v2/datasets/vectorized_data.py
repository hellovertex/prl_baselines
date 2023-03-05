import glob
import logging
import multiprocessing
import os
import re
import time
from functools import partial
from pathlib import Path
from typing import List, Type, Optional

import click
import numpy as np
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env
from tqdm import tqdm

from prl.baselines.supervised_learning.v2.datasets.dataset_options import (
    DatasetOptions,
    ActionGenOption)
from prl.baselines.supervised_learning.v2.datasets.persistent_storage import \
    PersistentStorage
from prl.baselines.supervised_learning.v2.datasets.raw_data import (
    TopPlayerSelector,
    RawData)
from prl.baselines.supervised_learning.v2.datasets.tmp import EncoderV2
from prl.baselines.supervised_learning.v2.fast_hsmithy_parser import \
    ParseHsmithyTextToPokerEpisode
from prl.baselines.supervised_learning.v2.poker_model import PokerEpisodeV2


class VectorizedData:
    def __init__(self,
                 dataset_options: DatasetOptions,
                 parser_cls: Type[ParseHsmithyTextToPokerEpisode],
                 top_player_selector: TopPlayerSelector,
                 storage: Optional[PersistentStorage] = None):
        self.opt = dataset_options
        self.parser_cls = parser_cls
        self.parser = parser_cls(nl=self.opt.nl)  # todo replace nl= with opt=
        self.top_player_selector = top_player_selector
        self.storage = storage if storage else PersistentStorage(self.opt)
        self.num_files_written_to_disk = 0

    @staticmethod
    def alias_player_rank_to_ingame_name(selected_player_names,
                                         filename:
                                         str) -> List[str]:
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
                                      encoder_cls: Type[EncoderV2],
                                      env: Optional[AugmentObservationWrapper] = None,
                                      use_multiprocessing: bool = False):
        # todo: implement
        raise NotImplementedError

    def _generate_vectorized_hand_histories(self,
                                            filename=None,
                                            encoder_cls=None,
                                            selected_player_names=None) -> str:
        # the env will be re-initialized with each hand in hand-histories, stacks and
        # blinds will be read from hand-history, so it does not matter what we provide
        # here
        dummy_env = init_wrapped_env(AugmentObservationWrapper,
                                     [5000 for _ in range(6)],
                                     blinds=(25, 50),
                                     multiply_by=1)
        encoder = encoder_cls(dummy_env)
        # single files, pretty large
        if not os.path.exists(self.opt.dir_vectorized_data):
            # filename to playername to one selected_players
            selected_players = self.alias_player_rank_to_ingame_name(
                selected_player_names, filename)
            episodesV2 = self.parser.parse_file(filename)
            training_data, labels = self.encode_episodes(episodesV2,
                                                         encoder,
                                                         selected_players)
            self.storage.vectorized_data_to_disk(training_data, labels,
                                                 encoder.feature_names)
            # todo: deprecate new_txt_to_vector_encoder and make tmp.EncoderV2
            #  encoder V2 the new one
        else:
            logging.info(f"Skipping encoding of hand histories, because they already "
                         f"exist at \n{self.opt.dir_vectorized_data}")
        return f"Success: encoded {filename}..."

    def _generate_player_pool_data(self,
                                   selected_player_names: List[str],
                                   files: List[str],
                                   encoder_cls: Type[EncoderV2],
                                   use_multiprocessing: bool = False):
        if use_multiprocessing:
            logging.info('Starting handhistory encoding using multiprocessing...')
            gen_fn = partial(self._generate_vectorized_hand_histories,
                             encoder_cls=encoder_cls,
                             selected_player_names=selected_player_names)
            assert len(files) < 101, f'Dont use multiprocessing for more than Top 100 ' \
                                     f'players.'
            start = time.time()
            p = multiprocessing.Pool()
            for x in p.imap_unordered(gen_fn, files):
                logging.info(x + f'. Took {time.time() - start} seconds\n')
            logging.info(f'*** Finished job after {time.time() - start} seconds. ***')
            p.close()
        else:
            logging.info('Starting handhistory encoding without multiprocessing...')
            for filename in files:
                self._generate_vectorized_hand_histories(
                    encoder_cls=encoder_cls,
                    selected_player_names=selected_player_names,
                    filename=filename)

    def _make_missing_data(self):
        # extracts hand histories for Top M players,
        # where M=self.opt.num_top_players-missing
        RawData(self.opt, self.top_player_selector).generate()

    def generate(self,
                 encoder_cls: Type[EncoderV2] = EncoderV2,
                 use_multiprocessing=False):

        if not self.opt.exists_raw_data_for_all_selected_players():
            self._make_missing_data()

        filenames = glob.glob(f'{self.opt.dir_raw_data_top_players}/**/*.txt',
                              recursive=True)

        selected_players = self.top_player_selector.get_top_n_players_min_showdowns(
            self.opt.num_top_players, self.opt.min_showdowns)
        logging.info(f"Encoding and vectorizing hand histories to .csv files for top "
                     f"{len(selected_players)} players. ")

        if self.opt.make_dataset_for_each_individual:
            return self._generate_per_selected_player(
                selected_player_names=list(selected_players.keys()),
                files=filenames,
                encoder_cls=encoder_cls,
                use_multiprocessing=use_multiprocessing)
        else:
            return self._generate_player_pool_data(
                selected_player_names=list(selected_players.keys()),
                files=filenames,
                encoder_cls=encoder_cls,
                use_multiprocessing=use_multiprocessing)


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
@click.option("--use_multiprocessing",
              default=False,
              type=bool,
              help="Whether to parallelize encoding of files per TopRanked Player. "
                   "Defaults to false.")
@click.option("--min_showdowns",
              default=5,
              type=int,
              help="Minimum number of showdowns required to be eligible for top player "
                   "ranking. Default is 5 for debugging. 5000 is recommended for real "
                   "data.")
def main(num_top_players,
         nl,
         make_dataset_for_each_individual,
         action_generation_option,
         use_multiprocessing,
         min_showdowns):
    # Assumes raw_data.py has been ran to download and extract hand histories.
    opt = DatasetOptions(
        num_top_players=num_top_players,
        nl=nl,
        make_dataset_for_each_individual=make_dataset_for_each_individual,
        action_generation_option=ActionGenOption(action_generation_option),
        min_showdowns=min_showdowns
    )
    parser_cls = ParseHsmithyTextToPokerEpisode
    selector = TopPlayerSelector(parser=parser_cls(nl))
    vectorized_data = VectorizedData(dataset_options=opt,
                                     parser_cls=parser_cls,
                                     top_player_selector=selector)
    vectorized_data.generate(use_multiprocessing=use_multiprocessing)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
