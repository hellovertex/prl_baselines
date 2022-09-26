from prl.baselines.supervised_learning.data_acquisition.hsmithy_parser import HSmithyParser


class PokerSnowieGenerator:
    def generate_database(self):
        """
        Generate https://www.pokersnowie.com/ databases
        """


class HandHistorySmithyToPokerSnowie(PokerSnowieGenerator):
    """ Translates databases from https://www.hhsmithy.com/ to https://www.pokersnowie.com/ databases
    Databases are .txt files in human readable format
    """

    def __init__(self, parser: HSmithyParser):
        self._parser = parser

    def generate_database(self, path_in, path_out):
        """Use prl.baselines.supervised_learning.data_acquisition.core.parser.PokerEpisode instances
        as intermediate translation tool.
        Args:
            path_in: Absolute path to hhsmithy .txt databases.
            path_out: Absolute path where PokerSnowie database result should be written to
        Returns:
             True, if the database was written successfully. False, if an Exception occured and no db was written.
        """
        # read .txt files

        # parse .txt files
        # PokerEpisode -> PokerSnowieEpisode
        # export to .txt file
        return False