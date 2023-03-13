import copy
import glob
import json
import logging
import multiprocessing
import os
import re
import time
from functools import partial
from pathlib import Path
from typing import List, Type, Optional, Generator

import click
import numpy as np
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env
from prl.environment.Wrappers.vectorizer import AgentObservationType
from tqdm import tqdm

from prl.baselines.evaluation.v2.dataset_stats import make_hud_stats_if_missing
from prl.baselines.supervised_learning.v2.datasets.dataset_config import (
    DatasetConfig,
    ActionGenOption)
from prl.baselines.supervised_learning.v2.datasets.persistent_storage import \
    PersistentStorage
from prl.baselines.supervised_learning.v2.datasets.raw_data import (
    TopPlayerSelector,
    RawData, make_raw_data_if_not_exists_already)
from prl.baselines.supervised_learning.v2.datasets.encoder import EncoderV2
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
    arg_min_showdowns,
    arg_hudstats
)


def alias_player_rank_to_ingame_name(selected_player_names,
                                     filename:
                                     str) -> List[str]:
    # map `01_raw/NL50/selected_players/PlayerRank008` to int(8)
    filename = Path(filename).name
    rank: int = int(re.search(r'\d+', filename).group())
    # map rank int to key index of top players
    for i, name in enumerate(selected_player_names):
        if i + 1 == rank:
            return [name]
    raise ValueError(f"No Player name has been found for file {filename}")


def encode_episodes(dataset_config,
                    episodesV2: Generator[PokerEpisodeV2, None, None],  #List[PokerEpisodeV2],
                    encoder: EncoderV2,
                    selected_players: List[str],
                    storage: PersistentStorage,
                    file_suffix: str,
                    max_episodes_per_file=25000):
    training_data, labels = None, None
    it = 0
    for ep in tqdm(episodesV2, total=max_episodes_per_file):
        it += 1
        try:
            observations, actions = encoder.encode_episode(
                ep,
                a_opt=dataset_config.action_generation_option,
                limit_num_players=5,
                selected_players=selected_players,
                use_hudstats=dataset_config.hudstats_enabled,
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
                # print(e)
                pass
            if it % max_episodes_per_file == 0:
                storage.vectorized_player_pool_data_to_disk(training_data,
                                                            labels,
                                                            encoder.feature_names,
                                                            file_suffix=file_suffix + str(
                                                                it))
                training_data, labels = None, None
    storage.vectorized_player_pool_data_to_disk(training_data,
                                                labels,
                                                encoder.feature_names,
                                                file_suffix=file_suffix + str(
                                                    it))
    training_data, labels = None, None


def generate_vectorized_hand_histories_from_chunks(files,
                                                   dataset_config: DatasetConfig,
                                                   parser_cls,
                                                   encoder_cls,
                                                   selected_player_names,
                                                   storage_cls) -> str:
    for filename in files[:-1]:
        generate_vectorized_hand_histories_from_file(filename,
                                                     dataset_config,
                                                     parser_cls,
                                                     encoder_cls,
                                                     selected_player_names,
                                                     storage_cls)
    return f"Success: encoded chunk {files}..."


def generate_vectorized_hand_histories_from_file(filename,
                                                 dataset_config: DatasetConfig,
                                                 parser_cls,
                                                 encoder_cls,
                                                 selected_player_names,
                                                 storage_cls,
                                                 agent_observation_mode=AgentObservationType.CARD_KNOWLEDGE) -> str:
    # make deepcopy so that multiprocessing does not share options object
    dataset_config = copy.deepcopy(dataset_config)
    # the env will be re-initialized with each hand in hand-histories, stacks and
    # blinds will be read from hand-history, so it does not matter what we provide
    # here
    dummy_env = init_wrapped_env(AugmentObservationWrapper,
                                 [5000 for _ in range(6)],
                                 blinds=(25, 50),
                                 multiply_by=1,
                                 agent_observation_mode=agent_observation_mode)
    encoder = encoder_cls(dummy_env)
    parser = parser_cls(dataset_config=dataset_config)
    storage = storage_cls(dataset_config)
    selected_players = alias_player_rank_to_ingame_name(selected_player_names,
                                                        filename)
    if dataset_config.hudstats_enabled:
        lut_file = os.path.join(*[
            dataset_config.dir_player_summaries,
            Path(filename).stem,
            'hud_lookup_table.json'
        ])
        with open(lut_file, 'r') as file:
            dict_obj = json.load(file)
        encoder.lut = dict_obj
    episodesV2 = parser.parse_lazily(filename)
    encode_episodes(dataset_config,
                    episodesV2,
                    encoder,
                    selected_players,
                    storage=storage,
                    file_suffix=Path(filename).stem)
    return f"Success: encoded {filename}..."


class VectorizedData:
    def __init__(self,
                 dataset_config: DatasetConfig,
                 parser_cls: Type[ParseHsmithyTextToPokerEpisode],
                 top_player_selector: TopPlayerSelector,
                 storage: Optional[PersistentStorage] = None):
        self.opt = dataset_config
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
                                            suffix,
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
        if self.opt.hudstats_enabled:
            lut_file = os.path.join(*[
                self.opt.dir_player_summaries,
                Path(filename).stem,
                'hud_lookup_table.json'
            ])
            with open(lut_file, 'r') as file:
                dict_obj = json.load(file)
            encoder.lut = dict_obj
        # few files, pretty large, one per top player
        selected_players = alias_player_rank_to_ingame_name(
            selected_player_names, filename)
        episodesV2 = self.parser.parse_file(filename)
        encode_episodes(self.opt,
                        episodesV2,
                        encoder,
                        selected_players,
                        storage=self.storage,
                        file_suffix=suffix)
        return f"Success: encoded {filename}..."

    def _generate_player_pool_data(self,
                                   selected_player_names: List[str],
                                   files: List[str],
                                   encoder_cls: Type[EncoderV2],
                                   use_multiprocessing: bool = False,
                                   chunksize=1  # smaller means more parallelization,
                                   # use with multiprocessing to avoid stackoverflow
                                   ):
        if use_multiprocessing:
            logging.info('Starting handhistory encoding using multiprocessing...')
            gen_fn = partial(generate_vectorized_hand_histories_from_chunks,
                             dataset_config=self.opt,
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
                    filename=filename,
                    suffix=files[-1])

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


def make_vectorized_data_if_not_exists_already(dataset_config, use_multiprocessing):
    make_raw_data_if_not_exists_already(dataset_config)
    parser_cls = ParseHsmithyTextToPokerEpisode
    selector = TopPlayerSelector(parser=parser_cls(dataset_config=dataset_config))
    vectorized_data = VectorizedData(dataset_config=dataset_config,
                                     parser_cls=parser_cls,
                                     top_player_selector=selector)
    if dataset_config.hudstats_enabled:
        make_hud_stats_if_missing(dataset_config)
    vectorized_data.generate_missing(use_multiprocessing=use_multiprocessing)


@click.command()
@arg_num_top_players
@arg_nl
@arg_from_gdrive_id
@arg_make_dataset_for_each_individual
@arg_action_generation_option
@arg_use_multiprocessing
@arg_min_showdowns
@arg_hudstats
def main(num_top_players,  # see click command main_raw_data
         nl,  # see click command main_raw_data
         from_gdrive_id,  # see click command main_raw_data
         make_dataset_for_each_individual,
         action_generation_option,
         use_multiprocessing,
         min_showdowns,
         hudstats):
    dataset_config = DatasetConfig(
        num_top_players=num_top_players,
        nl=nl,
        from_gdrive_id=from_gdrive_id,
        make_dataset_for_each_individual=make_dataset_for_each_individual,
        action_generation_option=ActionGenOption(action_generation_option),
        min_showdowns=min_showdowns,
        hudstats_enabled=hudstats
    )
    make_vectorized_data_if_not_exists_already(dataset_config, use_multiprocessing)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()  # skipped if exits_already
