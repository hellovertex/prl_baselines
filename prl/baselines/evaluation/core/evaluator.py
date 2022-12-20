from typing import List, Dict, Optional

from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode


class PokerExperimentEvaluator:
    """Base class for experiment evaluators"""

    # Note to self:
    # I think this might be useful for my repo `prl_reinforce` too,
    # so I made this class a Baseclass to keep it in the back of my head
    # that I will potentially refactor this out of `prl_baselines`
    def evaluate(self,
                 game_episodes: List[PokerEpisode],
                 eval_config: Optional[Dict]):  # todo -> ExperimentEvaluation
        """Evaluates the given poker episodes using metrics from eval_config.
        Under Construction"""
        raise NotImplementedError
