from prl.baselines.agents.eval.core.experiment import PokerExperiment


DEFAULT_DATE = "2022-12-24"
DEFAULT_VARIANT = "HUNL"
DEFAULT_CURRENCY = "$"


class PokerExperimentEvaluator:
    """Base class for experiment evaluators"""
    # Note to self:
    # I think this might be useful for my repo `prl_reinforce` too,
    # so I made this class a Baseclass to keep it in the back of my head
    # that I will potentially refactor this out of `prl_baselines`
    def evaluate(self, experiment: PokerExperiment):
        """Executes the given experiment. Under Construction"""
        raise NotImplementedError
