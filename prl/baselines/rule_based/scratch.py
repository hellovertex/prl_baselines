SB = 0
BB = 1

FOLD = 0
CHECK_CALL = 1
RAISE = 2
class BaseAgent:
    def act(self, obs):
        """ Takes observation vector of shape (563, ) and returns an action Tuple for prl_environment. """
        raise NotImplementedError


class TAGAgent:
    """Tight aggressive ruleset. """
    def __init__(self, starting_stack_sizes, blind_sizes):
        self._starting_stack_sizes = starting_stack_sizes
        self._blind_sizes = blind_sizes
        self._position = None  # [UTG, MP, CU, BTN, SB, BB]
        self._hand_cards = [[], []]  # [Rank, Suit] x N_HAND_CARDS
        # self._action_history = ActionHistory # action history of this round for all players

    def can_open_raise(stage, ranges):
        # use self._position
        raise NotImplementedError

    def open_raise(self):
        # can adjust to 2.5 - 3.5
        return (RAISE, 3 * self._blind_sizes[BB])

    def act(self, obs):
        # ranges = estimate_opponent_ranges()  # uses self._action_history to get ranges
        # stage = get_stage(obs)
        # if self.can_open_raise(stage, ranges):
        #   return maybe_open_raise()
        # if self.can_3bet(stage, ranges)
        #   return maybe_3bet()
        pass