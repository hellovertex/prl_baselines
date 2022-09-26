import glob

from prl.baselines.supervised_learning.data_acquisition.hsmithy_parser import HSmithyParser


class PokerSnowieGenerator:
    def generate_database(self, *args, **kwargs):
        """
        Generate https://www.pokersnowie.com/ databases
        """


class HandHistorySmithyToPokerSnowie(PokerSnowieGenerator):
    """ Translates databases from https://www.hhsmithy.com/ to https://www.pokersnowie.com/ databases
    Databases are .txt files in human readable format
    """

    def __init__(self, parser: HSmithyParser):
        self._parser = parser

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
        smithy_episodes = []
        snowie_episodes = []
        for f in filenames:
            for hand in self._parser.parse_file(f):
                smithy_episodes.append(hand)
            if len(smithy_episodes) > n_out_episodes_per_file:
                # PokerEpisode -> PokerSnowieEpisode

                # export to .txt file
                smithy_episodes = []
                snowie_episodes = []
        return True