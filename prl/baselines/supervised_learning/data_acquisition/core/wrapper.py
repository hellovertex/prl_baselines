class Wrapper:

    def __init__(self, env):
        """
        Args:
            env:   The environment instance to be wrapped
        """
        self.env = env

    def reset(self, config):
        """Reset the environment with a new config.
        Signals environment handlers to reset and restart the environment using
        a config dict.
        Args:
          config: dict, specifying the parameters of the environment to be
            generated. May contain state_dict to generate a deterministic environment.
        Returns:
          observation: A dict containing the full observation state.
        """
        raise NotImplementedError("Not implemented in Abstract Base class")

    def step(self, action):
        """Take one step in the game.
        Args:
          action: object, mapping to an action taken by an agent.
        Returns:
          observation: object, Containing full observation state.
          reward: float, Reward obtained from taking the action.
          done: bool, Whether the game is done.
          info: dict, Optional debugging information.
        Raises:
          AssertionError: When an illegal action is provided.
        """
        raise NotImplementedError("Not implemented in Abstract Base class")


class WrapperPokerRL(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._player_hands = []

    def reset(self, config=None):
        """
        Resets the state of the game to the standard beginning of the episode. If specified in the args passed,
        stack size randomization is applied in the new episode. If deck_state_dict is not None, the cards
        and associated random variables are synchronized FROM the given environment, so that when .step() is called on
        each of them, they produce the same result.

        Args:
            config["deck_state_dict"]:      Optional.
                                            If an instance of a PokerEnv subclass is passed, the deck, holecards, and
                                            board in this instance will be synchronized from the handed env cls.
        """
        # assert config.get('deck_state_dict')
        self._before_reset(config)
        deck_state_dict = None
        if config is not None:
            deck_state_dict = config['deck_state_dict']
        env_obs, rew_for_all_players, done, info = self.env.reset(deck_state_dict=deck_state_dict)
        if not self._player_hands:
            for i in range(self.env.N_SEATS):
                self._player_hands.append(self.env.get_hole_cards_of_player(i))
        # todo move this to proper location
        self._after_reset()

        return self._return_obs(env_obs=env_obs, rew_for_all_players=rew_for_all_players, done=done, info=info)

    def step(self, action):
        """
        Steps the environment from an action of the natural action representation to the environment.

        Returns:
            obs, reward, done, info
        """

        # callbacks in derived class
        self._before_step(action)

        # step environment
        env_obs, rew_for_all_players, done, info = self.env.step(action)

        self._after_step(action)
        # call get_current_obs of derived class
        return self._return_obs(env_obs=env_obs, rew_for_all_players=rew_for_all_players, done=done, info=info)

    def step_from_processed_tuple(self, action):
        """
        Steps the environment from a tuple (action, num_chips,).

        Returns:
            obs, reward, done, info
        """
        return self.step(action)

    def step_raise_pot_frac(self, pot_frac):
        """
        Steps the environment from a fractional pot raise instead of an action as usually specified.

        Returns:
            obs, reward, done, info
        """
        processed_action = (2, self.env.get_fraction_of_pot_raise(
            fraction=pot_frac, player_that_bets=self.env.current_player))

        return self.step(processed_action)

    def _return_obs(self, rew_for_all_players, done, info, env_obs=None):
        return self.get_current_obs(env_obs=env_obs), rew_for_all_players, done, info

    # _______________________________ Override to augment observation ________________________________

    def _before_step(self, action):
        raise NotImplementedError

    def _before_reset(self, config):
        raise NotImplementedError

    def _after_step(self, action):
        raise NotImplementedError

    def _after_reset(self):
        raise NotImplementedError

    def get_current_obs(self, env_obs):
        raise NotImplementedError

    # Can add additional callbacks here if necessary...
