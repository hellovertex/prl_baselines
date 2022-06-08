#
class BaseAgent:
    def act(self, obs):
        """ Takes observation vector of shape (563, ) and returns an action Tuple for prl_environment. """
        raise NotImplementedError


class TAGAgent:
    """Tight aggressive ruleset. """

    def act(self, obs):
        pass
