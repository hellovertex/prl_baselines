from prl.baselines.agents.core.base_agent import Agent


class BaselineAgent(Agent):
    def reset(self, config):
        """Not needed, keep for consistency with interface"""
        pass

    def __init__(self, config, *args, **kwargs):
        """Wrapper around rllib policy of our baseline agent obtained from supervised learning of game logs"""
        super().__init__(config, *args, **kwargs)
        self._rllib_policy = config['rllib_policy']

    @property
    def policy(self):
        return self._rllib_policy

    def act(self, observation):
        """Wraps rllib policy."""
        assert isinstance(observation, dict)
        return self._rllib_policy.compute_actions(observation)
