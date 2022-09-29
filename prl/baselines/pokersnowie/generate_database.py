import glob
from typing import List

from prl.baselines.pokersnowie.pokersnowie import SnowieConverter
from prl.baselines.supervised_learning.data_acquisition.hsmithy_parser import HSmithyParser


class PokerSnowieGenerator:
    def generate_database(self, *args, **kwargs):
        """
        Generate https://www.pokersnowie.com/ databases
        """


class HandHistorySmithyToPokerSnowie(PokerSnowieGenerator):
    """Translates databases from https://www.hhsmithy.com/ to https://www.pokersnowie.com/ databases.
    These are .txt files in human-readable format """

    def __init__(self, parser: HSmithyParser):
        self._parser = parser
        self.smithy_episodes = []
        self.snowie_episodes = []
        self._converter = SnowieConverter()
    # def _translate(self, smithy_episodes: List[PokerEpisode]) -> List[str]:

    def _parse_file(self, file_path, selected_players: List[str] = None):
        """Appends to self.snowie_episodes all poker episodes contained in the .txt file.
        If selected_players is passed, only games where these players participated will be returned"""
        # todo urgent skip parsing if no selected players is found in file
        for smithy_episode in self._parser.parse_file(file_path):
            self.smithy_episodes.append(smithy_episode)

    def _translate_episodes(self):
        for smithy_episode in self.smithy_episodes:
            # creates one episode per showdown player -- todo find better logic
            snowie_hands: list = self._converter.from_poker_episode(smithy_episode)
            [self.snowie_episodes.append(s) for s in snowie_hands]

    def _export_to_txt_file(self):
        # write self.snowie_episodes
        print(self.snowie_episodes)
        pass

    def generate_database(self, path_in, path_out, n_out_episodes_per_file):
        """Use prl.baselines.supervised_learning.data_acquisition.core.parser.PokerEpisode instances
        as intermediate translation tool.
        Args:
            path_in: Absolute path to hhsmithy .txt databases.
            path_out: Absolute path where PokerSnowie database result should be written to
            n_out_episodes_per_file: how many PokerSnowie hands should be written to a single .txt file
            (approximately)
        Returns:
             True, if the database was written successfully. False, if an Exception occurred and no db was written.
        """
        # read .txt files
        filenames = glob.glob(path_in.__str__() + '/*.txt', recursive=False)
        # parse .txt files
        # Note: potentially many 100k .txt files are present, which is why we parse them
        # one by one
        for f in filenames:
            # populate self.smithy_episodes
            self._parse_file(f)
            # consume self.smithy_episodes and create PokerSnowie database from them
            if len(self.smithy_episodes) > n_out_episodes_per_file:
                self._translate_episodes()  # convert to PokerSnowieEpisodes
                self._export_to_txt_file()  # write PokerSnowie database to disk
                self.smithy_episodes = []
                self.snowie_episodes = []
        return True
