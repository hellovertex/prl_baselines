import random

from prl.environment.Wrappers.base import ActionSpace


class RandomAgent:
    def __init__(self):
        self.n_actions = len(ActionSpace)

    def act(self, *args, **kwargs):
        return random.randint(0, 2)


class DummyAgentFold:
    def act(self, *args, **kwargs):
        return ActionSpace.FOLD


class DummyAgentCall:
    def act(self, *args, **kwargs):
        return ActionSpace.CHECK_CALL


class DummyAgentAllIn:
    def act(self, *args, **kwargs):
        return ActionSpace.RAISE_ALL_IN
