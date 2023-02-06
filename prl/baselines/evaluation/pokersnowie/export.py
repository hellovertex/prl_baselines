import ast
import glob
import os.path
from pathlib import Path
from typing import List, Optional, TypeVar, Union, Dict

from prl.baselines.evaluation.core.experiment import PokerExperiment
from prl.baselines.evaluation.experiment_runner import PokerExperimentRunner
from prl.baselines.evaluation.pokersnowie.converter_888 import Converter888
from prl.baselines.evaluation.pokersnowie.core.converter import PokerSnowieConverter, SnowieEpisode
from prl.baselines.evaluation.pokersnowie.core.db_generator import PokerSnowieExporteur
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode
from prl.baselines.supervised_learning.data_acquisition.hsmithy_parser import HSmithyParser

POKER_SNOWIE_CONVERTER_INSTANCE = TypeVar('POKER_SNOWIE_CONVERTER_INSTANCE', bound=PokerSnowieConverter)


class PersistentStorage:
    def _write(self, episodes, path_out, filename):
        if not os.path.exists(path_out):
            os.makedirs(os.path.abspath(path_out))
        with open(os.path.join(path_out, filename), 'a+') as f:
            for e in episodes:
                f.write(e)

    def export_to_text_file(self,
                            snowie_episodes: List[SnowieEpisode],
                            path_out: str,
                            file_suffix: Optional[str] = None,
                            max_episodes_per_file: int = 500
                            ):
        write_buffer = []
        n_files_written = 0
        for i, e in enumerate(snowie_episodes):
            if (i + 1) % max_episodes_per_file == 0:
                # flush and write
                self._write(write_buffer,
                            path_out,
                            filename='snowie'+f'_{n_files_written}.txt')
                n_files_written += 1
                write_buffer = []
            write_buffer.append(e)
        self._write(write_buffer,
                    path_out,
                    filename='snowie' + f'_{n_files_written}.txt')


class HandHistorySmithyToPokerSnowie(PokerSnowieExporteur):
    """Translates databases from https://www.hhsmithy.com/ to https://www.pokersnowie.com/ databases.
    These are .txt files in human-readable format """

    def __init__(self,
                 parser: HSmithyParser,
                 converter: POKER_SNOWIE_CONVERTER_INSTANCE,
                 storage: Optional[PersistentStorage]):
        self._parser = parser
        self.smithy_episodes = []
        self.snowie_episodes = []
        self._path_out = None
        self._converter = converter
        self._storage = storage if storage is not None else PersistentStorage()

    # def _translate(self, smithy_episodes: List[PokerEpisode]) -> List[str]:
    @staticmethod
    def _filter_players(smithy_episode: PokerEpisode, filter_by: Optional[List[str]]):
        if not filter_by:
            return True, None
        showdown_players = [p.name for p in smithy_episode.showdown_hands]
        for player in filter_by:
            if player in showdown_players:
                return True, player
        return False, None

    def _parse_file(self, file_path, filter_by: Optional[List[str]]):
        """Appends to self.snowie_episodes all poker episodes contained in the .txt file.
        If selected_players is passed, only games where these players participated will be returned"""
        for smithy_episode in self._parser.parse_file(file_path):
            proceed, player_name = self._filter_players(smithy_episode, filter_by)
            if proceed:
                self.smithy_episodes.append((smithy_episode, [player_name]))

    def _translate_episodes(self):
        for smithy_episode, hero_names in self.smithy_episodes:
            # creates one episode per showdown player
            snowie_hands: list = self._converter.from_poker_episode(smithy_episode, hero_names=hero_names)
            [self.snowie_episodes.append(s) for s in snowie_hands]

    @staticmethod
    def _get_selected_players(from_file):
        with open(from_file, "r") as data:
            player_dict = ast.literal_eval(data.read())
            return list(player_dict.keys())

    def generate_database(self, path_in, path_out, n_out_episodes_per_file, selected_players_file=None):
        """Use prl.baselines.supervised_learning.data_acquisition.core.parser.PokerEpisode instances
        as intermediate translation tool.
        Args:
            path_in: Absolute path to hhsmithy .txt databases.
            path_out: Absolute path where PokerSnowie database result should be written to
            n_out_episodes_per_file: how many PokerSnowie hands should be written to a single .txt file
            (approximately)
            selected_players_file:
        Returns:
             True, if the database was written successfully. False, if an Exception occurred and no db was written.
        """
        self._path_out = path_out

        # read .txt files
        filenames = glob.glob(path_in.__str__() + '/*.txt', recursive=False)
        # maybe get list of players which we want to filter by
        selected_players = None
        if selected_players_file:
            selected_players = self._get_selected_players(selected_players_file)
        # parse .txt files
        # Note: potentially many 100k .txt files are present, which is why we parse them
        # one by one
        n_files_parsed = 0
        for f in filenames:
            # populate self.smithy_episodes
            self._parse_file(f, filter_by=selected_players)
            # consume self.smithy_episodes and create PokerSnowie database from them
            if len(self.smithy_episodes) > n_out_episodes_per_file:
                self._translate_episodes()  # convert to PokerSnowieEpisodes
                self._storage.export_to_text_file(self.snowie_episodes,
                                                  self._path_out,
                                                  str(n_files_parsed))  # write PokerSnowie database to disk
                n_files_parsed += 1
                self.smithy_episodes = []
                self.snowie_episodes = []
        self._storage.export_to_text_file(self.snowie_episodes,
                                          self._path_out,
                                          str(n_files_parsed))
        return True


class PokerExperimentToPokerSnowie(PokerSnowieExporteur):
    def __init__(self,
                 converter=None,
                 experiment_runner=None,
                 storage: Optional[PersistentStorage] = None):
        self._converter = converter if converter is not None else Converter888()
        self._runner = experiment_runner if experiment_runner is not None else PokerExperimentRunner()
        self._storage = storage if storage is not None else PersistentStorage()

    def generate_database(self,
                          path_out: Union[str, Path],
                          experiment: PokerExperiment,
                          max_episodes_per_file=500,
                          hero_names: Optional[List[str]] = None,
                          verbose=False):
        # execute Experiment to generate list of poker episodes
        poker_episodes = self._runner.run(experiment, verbose=verbose)
        snowie_episodes = []
        # parse list of poker episodes to snowie-formatted string
        for ep in poker_episodes:
            # showdown_eps is a list of the same episode from different angles relative to observer
            showdown_eps = self._converter.from_poker_episode(ep, hero_names)
            for observer_relative in showdown_eps:
                snowie_episodes.append(observer_relative)
        # write snowie-formatted string to text file
        self._storage.export_to_text_file(snowie_episodes=snowie_episodes,
                                          path_out=path_out)
        return self
    def summary(self) -> Dict[str, List]:
        # res = self._runner
        return self._runner.winnings
