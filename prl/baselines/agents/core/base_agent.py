class Agent(object):
    """Agent interface.
    All concrete implementations of an Agent should derive from this interface
    and implement the method stubs.
    ```python
    class MyAgent(Agent):
      ...
    agents = [MyAgent(config) for _ in range(players)]
    while not done:
      ...
      for agent_id, agent in enumerate(agents):
        action = agent.act(observation)
        if obs.current_player == agent_id:
          assert action is not None
        else
          assert action is None
      ...
    ```
    """

    def __init__(self, config, *args, **kwargs):
        r"""Initialize the agent.
        Args:
          config: dict, With parameters for the game.
          *args: Optional arguments
          **kwargs: Optional keyword arguments.
        Raises:
          AgentError: Custom exceptions.
        """
        self._config = config

    def reset(self, config):
        r"""Reset the agent with a new config.
        Signals agent to reset and restart using a config dict.
        Args:
          config: dict, With parameters for the game.
        """
        raise NotImplementedError("Not implemeneted in abstract base class.")

    def act(self, observation):
        """Act based on an observation.
        Args:
          observation: dict, containing observation from the view of this agent.
            An example:
            - tba, see AugmentedObservationFeatureColumns
        Returns:
          action: dict, mapping to a legal action taken by this agent. The following
            actions are supported:
            - tba, see PokerEnv
        """
        raise NotImplementedError("Not implemented in Abstract Base class")
