import copy
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

from prl.baselines.supervised_learning.v2.datasets.dataset_config import (
    DatasetConfig,
    ActionGenOption)
from prl.baselines.supervised_learning.v2.datasets.persistent_storage import \
    PersistentStorage
from prl.baselines.supervised_learning.v2.datasets.raw_data import (
    TopPlayerSelector,
    RawData, make_raw_data_if_not_exists_already)
from prl.baselines.supervised_learning.v2.datasets.tmp import EncoderV2
from prl.baselines.supervised_learning.v2.fast_hsmithy_parser import \
    ParseHsmithyTextToPokerEpisode
from prl.baselines.supervised_learning.v2.poker_model import PokerEpisodeV2

from prl.baselines.supervised_learning.v2.datasets.dataset_config import (
    arg_num_top_players,
    arg_nl,
    arg_from_gdrive_id,
    arg_make_dataset_for_each_individual,
    arg_action_generation_option,
    arg_use_multiprocessing,
    arg_min_showdowns
)


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


def encode_episodes(dataset_options,
                    episodesV2: List[PokerEpisodeV2],
                    encoder: EncoderV2,
                    selected_players: List[str],
                    storage: PersistentStorage,
                    file_suffix: str,
                    max_lines_per_file=50000):
    training_data, labels = None, None
    it = 0
    for ep in tqdm(episodesV2):
        try:
            observations, actions = encoder.encode_episode(
                ep,
                a_opt=dataset_options.action_generation_option,
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
            if len(training_data) % max_lines_per_file == 0:
                storage.vectorized_player_pool_data_to_disk(training_data,
                                                            labels,
                                                            encoder.feature_names,
                                                            file_suffix=file_suffix+str(it))
                it += 1

    return training_data, labels


def generate_vectorized_hand_histories(files,
                                       dataset_options: DatasetConfig,
                                       parser_cls,
                                       encoder_cls,
                                       selected_player_names,
                                       storage_cls) -> str:
    # make deepcopy so that multiprocessing does not share options object
    dataset_options = copy.deepcopy(dataset_options)
    # the env will be re-initialized with each hand in hand-histories, stacks and
    # blinds will be read from hand-history, so it does not matter what we provide
    # here
    dummy_env = init_wrapped_env(AugmentObservationWrapper,
                                 [5000 for _ in range(6)],
                                 blinds=(25, 50),
                                 multiply_by=1)
    encoder = encoder_cls(dummy_env)
    parser = parser_cls(nl=dataset_options.nl)  # todo replace `nl` with dataset_options param
    storage = storage_cls(dataset_options)
    for filename in files[:-1]:
        selected_players = alias_player_rank_to_ingame_name(selected_player_names, filename)
        episodesV2 = parser.parse_file(filename)
        training_data, labels = encode_episodes(dataset_options,
                                                episodesV2,
                                                encoder,
                                                selected_players,
                                                storage=storage,
                                                file_suffix=files[-1])
        # else:
        #     logging.info(f"Skipping encoding of hand histories, because they already "
        #                  f"exist at \n{dataset_options.dir_vectorized_data}")
    return f"Success: encoded chunk {files}..."


class VectorizedData:
    def __init__(self,
                 dataset_options: DatasetConfig,
                 parser_cls: Type[ParseHsmithyTextToPokerEpisode],
                 top_player_selector: TopPlayerSelector,
                 storage: Optional[PersistentStorage] = None):
        self.opt = dataset_options
        self.parser_cls = parser_cls
        self.parser = parser_cls(self.opt)  # todo replace nl= with opt=
        self.top_player_selector = top_player_selector
        self.storage = storage if storage else PersistentStorage(self.opt)
        self.num_files_written_to_disk = 0

    def _generate_per_selected_player(self,
                                      selected_player_names: List[str],
                                      files: List[str],
                                      encoder_cls: Type[EncoderV2],
                                      env: Optional[AugmentObservationWrapper] = None,
                                      use_multiprocessing: bool = False):
        # todo: implement
        raise NotImplementedError

    def _generate_vectorized_hand_histories(self,
                                            filename,
                                            encoder_cls,
                                            selected_player_names) -> str:
        # the env will be re-initialized with each hand in hand-histories, stacks and
        # blinds will be read from hand-history, so it does not matter what we provide
        # here
        dummy_env = init_wrapped_env(AugmentObservationWrapper,
                                     [5000 for _ in range(6)],
                                     blinds=(25, 50),
                                     multiply_by=1)
        encoder = encoder_cls(dummy_env)
        # few files, pretty large, one per top player
        selected_players = alias_player_rank_to_ingame_name(
            selected_player_names, filename)
        episodesV2 = self.parser.parse_file(filename)
        training_data, labels = encode_episodes(self.opt,
                                                episodesV2,
                                                encoder,
                                                selected_players)
        self.storage.vectorized_player_pool_data_to_disk(training_data,
                                                         labels,
                                                         encoder.feature_names)
        return f"Success: encoded {filename}..."

    def _generate_player_pool_data(self,
                                   selected_player_names: List[str],
                                   files: List[str],
                                   encoder_cls: Type[EncoderV2],
                                   use_multiprocessing: bool = False,
                                   chunksize=5  # use with multiprocessing to avoid stackoverflow
                                   ):
        if use_multiprocessing:
            logging.info('Starting handhistory encoding using multiprocessing...')
            gen_fn = partial(generate_vectorized_hand_histories,
                             dataset_options=self.opt,
                             parser_cls=type(self.parser),
                             encoder_cls=encoder_cls,
                             selected_player_names=selected_player_names,
                             storage_cls=type(self.storage))
            assert len(files) < 101, f'Dont use multiprocessing for more than Top 100 ' \
                                     f'players.'
            chunks = []
            current_chunk = []
            i = 0
            for file in files:
                current_chunk.append(file)
                if (i + 1) % chunksize == 0:
                    chunks.append(current_chunk)
                    current_chunk = []
                i += 1
            # trick to avoid multiprocessing writes to same file
            for i, chunk in enumerate(chunks):
                chunk.append(f'{i}')
            start = time.time()
            p = multiprocessing.Pool()
            for x in p.imap_unordered(gen_fn, chunks):
                logging.info(x + f'. Took {time.time() - start} seconds\n')
            logging.info(f'*** Finished job after {time.time() - start} seconds. ***')
            p.close()
        else:
            # raise NotImplementedError
            logging.info('Starting handhistory encoding without multiprocessing...')
            for filename in files:
                self._generate_vectorized_hand_histories(
                    encoder_cls=encoder_cls,
                    selected_player_names=selected_player_names,
                    filename=filename)

    def generate_missing(self,
                         encoder_cls: Type[EncoderV2] = EncoderV2,
                         use_multiprocessing=False):
        filenames = glob.glob(f'{self.opt.dir_raw_data_top_players}/**/*.txt',
                              recursive=True)

        selected_players = self.top_player_selector.get_top_n_players_min_showdowns(
            self.opt.num_top_players, self.opt.min_showdowns)

        if self.opt.exists_vectorized_data_for_all_selected_players():
            logging.info("Vectorized Data exists already. Skipping Encoder Step.")
        else:
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


def make_vectorized_data_if_not_exists_already(num_top_players,  # see click command main_raw_data
                                               nl,  # see click command main_raw_data
                                               from_gdrive_id,  # see click command main_raw_data
                                               make_dataset_for_each_individual,
                                               action_generation_option,
                                               use_multiprocessing,
                                               min_showdowns):
    make_raw_data_if_not_exists_already(num_top_players, nl, from_gdrive_id)
    opt = DatasetConfig(
        num_top_players=num_top_players,
        nl=nl,
        from_gdrive_id=from_gdrive_id,
        make_dataset_for_each_individual=make_dataset_for_each_individual,
        action_generation_option=ActionGenOption(action_generation_option),
        min_showdowns=min_showdowns
    )
    parser_cls = ParseHsmithyTextToPokerEpisode
    selector = TopPlayerSelector(parser=parser_cls(dataset_config=opt))
    vectorized_data = VectorizedData(dataset_options=opt,
                                     parser_cls=parser_cls,
                                     top_player_selector=selector)
    vectorized_data.generate_missing(use_multiprocessing=use_multiprocessing)


@click.command()
@arg_num_top_players
@arg_nl
@arg_from_gdrive_id
@arg_make_dataset_for_each_individual
@arg_action_generation_option
@arg_use_multiprocessing
@arg_min_showdowns
def main(num_top_players,  # see click command main_raw_data
         nl,  # see click command main_raw_data
         from_gdrive_id,  # see click command main_raw_data
         make_dataset_for_each_individual,
         action_generation_option,
         use_multiprocessing,
         min_showdowns):
    make_vectorized_data_if_not_exists_already(num_top_players,
                                               nl,
                                               from_gdrive_id,
                                               make_dataset_for_each_individual,
                                               action_generation_option,
                                               use_multiprocessing,
                                               min_showdowns)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()  # skipped if exits_already
