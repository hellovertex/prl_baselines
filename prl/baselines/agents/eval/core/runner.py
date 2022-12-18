from typing import List

from prl.baselines.agents.eval.core.experiment import PokerExperiment
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode

DEFAULT_DATE = "2022-12-24"
DEFAULT_VARIANT = "HUNL"
DEFAULT_CURRENCY = "$"


class ExperimentRunner:
    """Base class for experiment runners"""
    # Note to self:
    # I think this might be useful for my repo `prl_reinforce` too,
    # so I made this class a Baseclass to keep it in the back of my head
    # that I will potentially refactor this out of `prl_baselines`
    def run(self, experiment: PokerExperiment) -> List[PokerEpisode]:
        """Executes the given experiment. Under Construction"""
        raise NotImplementedError
