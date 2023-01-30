"""
player_name | Agression Factor | Tightness | acceptance level | Agression Factor NN | tightness NN
--------------------------------------------------------------------------------------------------
Agression Factor (AF): #raises / #calls
Tightness: % hands played (not folded immediately preflop)
"""
import numpy as np
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as fts
from prl.environment.Wrappers.base import ActionSpace


# load vectorized observations
# In: ndarray or .txt
# none of the stats can be computed from single observatoins they all require avg
# however updating them can
class PlayerStats:

    def __init__(self, pname, *args, **kwargs):
        self.pname = pname
        self.vpip = 0
        self.pfr = 0
        self.tightness = 0
        self.threebet = 0
        self.cbet = 0
        self.af = 0
        self.num_checkcall_or_folds = 0
        self.num_bets_or_raises = 0
        self.hands_to_showdown = 0
        self.hands_played = 0
        self.hands_total = 0

    def big_blind_checked_preflop(self, obs, action):
        if action != ActionSpace.CHECK_CALL:
            return False
        if obs[fts.Total_to_call] <= obs[fts.Curr_bet_p0]:
            # Action was Check instead of Call
            return True
        return False

    def _update_vpip(self, obs, action):
        """Percentage of time player makes calls or raises before the flop."""
        if fts.Round_preflop:
            # if action is not fold and player is not big blind who checks, update vpip
            if not action == ActionSpace.FOLD:
                # as big blind, only increment vpip if we call a bet or raise
                if not self.big_blind_checked_preflop(obs, action):
                    # any other call or raise increments vpip
                    self.vpip = (self.vpip + 1) / self.hands_total

    def _update_af(self, obs, action):
        if action < ActionSpace.RAISE_MIN_OR_3BB:
            self.num_checkcall_or_folds += 1
        else:
            self.num_bets_or_raises += 1
        self.af = self.num_bets_or_raises / self.num_checkcall_or_folds

    def _update_pfr(self, obs, action):
        pass

    def _update_cbet(self, obs, action):
        pass

    def _update_3bet(self, obs, action):
        pass

    def update_stats(self, obs: np.ndarray, action: int):
        self.hands_total += 1
        self._update_vpip(obs, action)
        self._update_af(obs, action)
        self._update_pfr(obs, action)
        self._update_cbet(obs, action)
        self._update_3bet(obs, action)

    def reset(self):
        """If necessary, we can reset all stats to 0 here, but I dont see where this could
        be useful"""
        pass

    def to_dict(self):
        return {'vpip': self.vpip,
                'pfr': self.pfr,
                'tightness': self.tightness,
                'threebet': self.threebet,
                'cbet': self.cbet,
                'af': self.af,
                'hands_to_showdown': self.hands_to_showdown,  # compute wtsd
                'hands_played': self.hands_played,
                'hands_total': self.hands_total,
                }

# [ ] 1. go into encoder and start building stats table while encoding from_dir = player_data with network loaded
# [x] 2. fix win_prob < ? condition - fix pot odds && fix whatif total_to_call=0
# --> if win_prob < total_to_call / (obs[cols.Pot_amt] + total_to_call):
# i: pot odds) this is correct. the player should win more than 1/(3+1) = 25% of the time.
# i.e. if he wins less often he shpould fold
# ii: whatif totaltocall=0) then we want to sometimes check sometimes raise
# we first see if we can raise which is determined by the acceptance level so it is a perfect hyperparameter
# to tune the AF and tightness of the baseline  % todo put this in the .tex file -- its our contrib to
#  todo: have made this a hyperparam to evolve from the game logs baseline paper

# fold probability is marginalized by tightness and by acceptance but we can fix the tightness to the players
# so the only remaining parameter is acceptance level. Again, todo move this to .tex

# Notes

# jeez i dont need the action it can be 99 in the replay buffer nobody cares
# the network is only trained with observations. the action it did itself it knows
