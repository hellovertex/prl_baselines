from prl.baselines.agents.core.base_agent import Agent, RllibAgent
from prl.baselines.agents.policies import StakeLevelImitationPolicy, AlwaysCallingPolicy


class CallingStation(RllibAgent):
    def reset(self, config):
        """Not needed, keep for consistency with interface"""
        pass

    def __init__(self, config, *args, **kwargs):
        """Wrapper around rllib policy of our baseline agent obtained from supervised learning of game logs"""
        super().__init__(config, *args, **kwargs)
        self._rllib_policy = AlwaysCallingPolicy(
            observation_space=config['observation_space'],
            action_space=config['action_space'],
            config=config['policy_config']
        )

    def act(self, observation):
        """Wraps rllib policy."""
        assert isinstance(observation, dict)
        return self._rllib_policy.compute_actions(observation)


class StakePlayerImitator(RllibAgent):
    def reset(self, config):
        """Not needed, keep for consistency with interface"""
        pass

    def __init__(self, config, *args, **kwargs):
        """Wrapper around rllib policy of our baseline agent obtained from supervised learning of game logs"""
        super().__init__(config, *args, **kwargs)
        self._rllib_policy = StakeLevelImitationPolicy(
            observation_space=config['observation_space'],
            action_space=config['action_space'],
            config=config['policy_config']
        )

    def act(self, observation):
        """Wraps rllib policy."""
        assert isinstance(observation, dict)
        return self._rllib_policy.compute_actions(observation)
