import numpy as np
from prl.environment.Wrappers.base import ActionSpace
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as fts


class HandCounter:
    def __init__(self):
        self.card_frequencies = np.zeros((13, 13))
        self.hands_played = np.zeros((13, 13))
        self.current_hand = None, None

    def parse_cards(self, obs):
        return 0, 0

    def hand_is_new(self, i, j):
        return not (self.current_hand[0] == i and self.current_hand[1] == j)

    def bb_checked_preflop(self, obs, action):
        assert obs[fts.Round_preflop]
        if action != ActionSpace.CHECK_CALL:
            return False
        if obs[fts.Total_to_call] <= obs[fts.Curr_bet_p0]:
            # Action was Check instead of Call
            return True
        return False

    def update_count(self, obs, action):
        """We count frequencies of possible starting hands played in a 13x13 matrix.
        We say a hand is played, if it is not immediately folded preflop.
        It can however be the case that an agent A decide to raise his JhJd but then someone
        else donkbets all in. If then agent A decides to fold we say the hand has been played,
        because A had the intention of playing it, but then got scared away.
        """
        i, j = self.parse_cards(obs)
        if self.hand_is_new(i, j):
            self.card_frequencies[i][j] += 1
            if obs[fts.Round_preflop]:
                # if preflop and action != FOLD and not bb_checked: increment card count
                if (action != ActionSpace.FOLD) or (not self.bb_checked_preflop(obs, action)):
                    self.hands_played[i][j] += 1
            self.current_hand = i, j