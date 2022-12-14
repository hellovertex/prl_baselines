import ast
import glob
from typing import List, Optional, TypeVar

from prl.baselines.pokersnowie.core.converter import PokerSnowieConverter
from prl.baselines.pokersnowie.core.db_generator import PokerSnowieExporteur
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode
from prl.baselines.supervised_learning.data_acquisition.hsmithy_parser import HSmithyParser

POKER_SNOWIE_CONVERTER_INSTANCE = TypeVar('POKER_SNOWIE_CONVERTER_INSTANCE', bound=PokerSnowieConverter)


class HandHistorySmithyToPokerSnowie(PokerSnowieExporteur):
    """Translates databases from https://www.hhsmithy.com/ to https://www.pokersnowie.com/ databases.
    These are .txt files in human-readable format """

    def __init__(self, parser: HSmithyParser, converter: POKER_SNOWIE_CONVERTER_INSTANCE):
        self._parser = parser
        self.smithy_episodes = []
        self.snowie_episodes = []
        self._path_out = None
        self._converter = converter

    # def _translate(self, smithy_episodes: List[PokerEpisode]) -> List[str]:
    @staticmethod
    def _filter_criteria_matched(smithy_episode: PokerEpisode, filter_by: Optional[List[str]]):
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
            proceed, player_name = self._filter_criteria_matched(smithy_episode, filter_by)
            if proceed:
                self.smithy_episodes.append((smithy_episode, [player_name]))

    def _translate_episodes(self):
        for smithy_episode, hero_names in self.smithy_episodes:
            # creates one episode per showdown player
            snowie_hands: list = self._converter.from_poker_episode(smithy_episode, hero_names=hero_names)
            [self.snowie_episodes.append(s) for s in snowie_hands]

    def _export_to_txt_file(self, n_written: int):
        # write self.snowie_episodes
        print(self.snowie_episodes)
        with open(self._path_out + rf'\snowiedb_{n_written}.txt', 'a') as f:
            for e in self.snowie_episodes:
                f.write(e)

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
        files_written = 0
        for f in filenames:
            # populate self.smithy_episodes
            self._parse_file(f, filter_by=selected_players)
            # consume self.smithy_episodes and create PokerSnowie database from them
            if len(self.smithy_episodes) > n_out_episodes_per_file:
                self._translate_episodes()  # convert to PokerSnowieEpisodes
                self._export_to_txt_file(files_written)  # write PokerSnowie database to disk
                files_written += 1
                self.smithy_episodes = []
                self.snowie_episodes = []
        self._export_to_txt_file(files_written)
        return True
