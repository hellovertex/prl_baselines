from typing import List

from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode

SnowieEpisode = str


class PokerSnowieConverter:
    """
    PokerSnowie software internally stores played hands as text files.
    Our internal representation of played hands is given by PokerEpisode - instances.

    Converter instances are supposed to convert
    a `prl.baselines.supervised_learning.data_acquisition.core.parser.PokerEpisode` - instance to .txt file
    for import in PokerSnowie. Different Formats can be used, e.g. PokerSnowieExportFormat or 888ExportFormat e
    """

    def from_poker_episode(self, episode: PokerEpisode, hero_names: List[str] = None) -> List[SnowieEpisode]:
        """Converts episode to string representation that can be imported from PokerSnowie if written to txt file."""
        raise NotImplementedError
