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
        # open raise given position and open raising range
        # 3-bet given position and opponents assumed open raising range
        # 4-bet / 5 bet given position and opponents assumed 3-betting range
        # cold calling given players left to act, and callers position -> and our hands equity
        #  given their open raising range, the more remaining players, the smaller our calling range,
        #   we can make this a function of num remaining players where we call we call with n hands on 1 rem.
        #    n-2 with 2 rem, etc...
        # if should_fold:
        # if should_call:
        # if should_raise:
        # ranges = estimate_opponent_ranges()  # uses self._action_history to get ranges
        # stage = get_stage(obs)
        # if self.can_open_raise(stage, ranges):
        #   return maybe_open_raise()
        # if self.can_3bet(stage, ranges)
        #   return maybe_3bet()
        pass