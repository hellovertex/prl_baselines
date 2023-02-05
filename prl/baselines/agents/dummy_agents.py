from prl.environment.Wrappers.base import ActionSpace


class DummyAgentFold:
    def act(self, *args, **kwargs):
        return ActionSpace.FOLD


class DummyAgentCall:
    def act(self, *args, **kwargs):
        return ActionSpace.CHECK_CALL


class DummyAgentAllIn:
    def act(self, *args, **kwargs):
        return ActionSpace.RAISE_ALL_IN
