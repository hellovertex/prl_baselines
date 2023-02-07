import json
import os
from pathlib import Path

import numpy as np
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as fts
from prl.environment.Wrappers.base import ActionSpace


class PlayerStats:
    # updates only if we have not already for this hand
    def __init__(self, pname, *args, **kwargs):
        self.pname = pname
        self.vpip = 0
        self.pfr = 0
        self.tightness = 0
        self.threebet = 0
        self.cbet = {'flop': 0.0,
                     'turn': 0.0,
                     'river': 0.0}
        self.af = 0
        self.num_immediate_folds = 0
        self.raises_preflop = 0
        self.raises_flop = 0
        self.raises_turn = 0
        self.raises_river = 0
        self.cbets_flop = 0
        self.cbets_turn = 0
        self.cbets_river = 0
        self.num_checkcall_or_folds = 0
        self.num_bets_or_raises = 0
        self.hands_to_showdown = 0
        self.hands_played = 0
        self.hands_total = 0
        self.total_can_three_bet = 0
        self.times_three_betted = 0
        self.vpip_updated_this_hand = False
        self.tightness_updated_this_hand = False
        self.pfr_updated_this_preflop = False
        self.cbet_flop_updated_this_hand = False
        self.cbet_turn_updated_this_hand = False
        self.cbet_river_updated_this_hand = False
        self.three_bet_updated_this_hand = False
        self.n_vpip = 0
        self.n_pfr = 0
        self.registered_folds = 0
        self.registered_calls = 0
        self.registered_raises = 0

    def big_blind_checked_preflop(self, obs, action):
        assert obs[fts.Round_preflop]
        if action != ActionSpace.CHECK_CALL:
            return False
        if obs[fts.Total_to_call] <= obs[fts.Curr_bet_p0]:
            # Action was Check instead of Call
            return True
        return False

    def _update_vpip(self, obs, action):
        """Percentage of time player makes calls or raises before the flop.
        The vpip is updated once per hand."""
        if self.vpip_updated_this_hand:
            return

        if obs[fts.Round_preflop]:
            # if action is not fold and player is not big blind who checks, update vpip
            if not action == ActionSpace.FOLD:
                # as big blind, only increment vpip if we call a bet or raise
                if not self.big_blind_checked_preflop(obs, action):
                    # any other call or raise increments vpip
                    self.n_vpip += 1
            self.vpip = self.n_vpip / self.hands_total
        self.vpip_updated_this_hand = True

    def _update_af(self, obs, action):
        """Agression factor: #(Bet + Raise) / #(Call, checking or folding).
        The af is updated with every action."""
        if action < ActionSpace.RAISE_MIN_OR_3BB:
            self.num_checkcall_or_folds += 1
        else:
            self.num_bets_or_raises += 1
        try:
            self.af = self.num_bets_or_raises / self.num_checkcall_or_folds
        except ZeroDivisionError:
            if self.num_bets_or_raises > 0:
                self.af = self.num_bets_or_raises
            else:
                self.af = 0

    def _update_pfr(self, obs, action):
        """Preflop Bets/Raises. The pfr is updated once per hand."""
        # only update pfr if we have not already for this hand
        if self.pfr_updated_this_preflop:
            return

        if obs[fts.Round_preflop]:
            if action >= ActionSpace.RAISE_MIN_OR_3BB:
                self.n_pfr += 1
            self.pfr = self.n_pfr / self.hands_total
            self.pfr_updated_this_preflop = True

    def _update_tightness(self, obs, action):
        """Percentage of hands played. tightness = 0.9 means player plays 90% of hands
        A hand is played if it is not folded immediately preflop.
        The tightness is updated once per hand. """
        if self.tightness_updated_this_hand:
            return
        if obs[fts.Round_preflop] and self.is_first_action:
            if action == ActionSpace.FOLD:
                self.num_immediate_folds += 1
        self.tightness = 1 - (self.num_immediate_folds / self.hands_total)
        self.tightness_updated_this_hand = True

    @staticmethod
    def _player_has_not_acted_in_flop(obs):
        return not (obs[fts.Flop_player_0_action_0_what_0] or
                    obs[fts.Flop_player_0_action_0_what_1] or
                    obs[fts.Flop_player_0_action_0_what_2])

    def _update_cbet_flop(self, obs, action):
        if self.cbet_flop_updated_this_hand:
            return
        player_raised_preflop = obs[fts.Preflop_player_0_action_0_what_2] or obs[fts.Preflop_player_0_action_1_what_2]
        if obs[fts.Round_flop]:
            # only update cbet stats on first move in flop
            if self._player_has_not_acted_in_flop(obs):
                if player_raised_preflop:
                    self.raises_preflop += 1
                    if action >= ActionSpace.RAISE_MIN_OR_3BB:
                        self.cbets_flop += 1
                    self.cbet['flop'] = self.cbets_flop / self.raises_preflop
            self.cbet_flop_updated_this_hand = True

    @staticmethod
    def _player_has_not_acted_in_turn(obs):
        return not (obs[fts.Turn_player_0_action_0_what_0] or
                    obs[fts.Turn_player_0_action_0_what_1] or
                    obs[fts.Turn_player_0_action_0_what_2])

    def _update_cbet_turn(self, obs, action):
        if self.cbet_turn_updated_this_hand:
            return
        player_raised_flop = obs[fts.Flop_player_0_action_0_what_2] or obs[fts.Flop_player_0_action_1_what_2]
        if obs[fts.Round_turn]:
            # only update cbet stats on first move in flop
            if self._player_has_not_acted_in_turn(obs):
                if player_raised_flop:
                    self.raises_flop += 1
                    if action >= ActionSpace.RAISE_MIN_OR_3BB:
                        self.cbets_turn += 1
                    self.cbet['turn'] = self.cbets_turn / self.raises_flop
            self.cbet_turn_updated_this_hand = True

    @staticmethod
    def _player_has_not_acted_in_river(obs):
        return not (obs[fts.River_player_0_action_0_what_0] or
                    obs[fts.River_player_0_action_0_what_1] or
                    obs[fts.River_player_0_action_0_what_2])

    def _update_cbet_river(self, obs, action):
        if self.cbet_river_updated_this_hand:
            return
        player_raised_turn = obs[fts.Turn_player_0_action_0_what_2] or obs[fts.Turn_player_0_action_1_what_2]
        if obs[fts.Round_river]:
            if self._player_has_not_acted_in_river(obs):
                if player_raised_turn:
                    self.raises_turn += 1
                    if action >= ActionSpace.RAISE_MIN_OR_3BB:
                        self.cbets_river += 1

                    self.cbet['river'] = self.cbets_river / self.raises_turn
            self.cbet_river_updated_this_hand = True

    def _update_cbet(self, obs, action):
        """Continuation Bet (Cbet): If player raised in previous round, percentage of times
        the player opens with a bet in the next round.
        Cbets are updated once per hand. """
        self._update_cbet_flop(obs, action)
        self._update_cbet_turn(obs, action)
        self._update_cbet_river(obs, action)

    def _update_3bet(self, obs, action):
        """3bet: First re-raise Pre-flop"""
        # A 3bet is present when the following conditions are met
        # i) Exactly one opponent raised as their _first_ action and
        # ii) None of the opponents raised as their _second_ action
        # iii) This is our first preflop action, and it is a raise
        exactly_one_opponent_raised_as_their_first_action = sum([
            obs[fts.Preflop_player_1_action_0_what_2],
            obs[fts.Preflop_player_2_action_0_what_2],
            obs[fts.Preflop_player_3_action_0_what_2],
            obs[fts.Preflop_player_4_action_0_what_2],
            obs[fts.Preflop_player_5_action_0_what_2],
        ]) == 1
        none_of_the_opponents_raised_as_their_second_action = sum([
            obs[fts.Preflop_player_1_action_1_what_2],
            obs[fts.Preflop_player_2_action_1_what_2],
            obs[fts.Preflop_player_3_action_1_what_2],
            obs[fts.Preflop_player_4_action_1_what_2],
            obs[fts.Preflop_player_5_action_1_what_2],
        ]) == 0
        if exactly_one_opponent_raised_as_their_first_action and none_of_the_opponents_raised_as_their_second_action:
            if self.is_first_action:
                self.total_can_three_bet += 1
                if action >= ActionSpace.RAISE_MIN_OR_3BB:
                    self.times_three_betted += 1
                self.threebet = self.times_three_betted / self.total_can_three_bet
                self.three_bet_updated_this_hand = True

    def new_hands_dealt(self, obs, action):
        """Consider using this instead of is_new_hand parameter."""
        # a new hand is dealt when three conditions are met in obs
        # i) round == preflop
        # ii) pot == sb + bb
        # iii) none of the players have folded yet
        pass

    def update_stats(self, obs: np.ndarray, action: int, is_new_hand: bool):
        self.is_new_hand = is_new_hand
        if is_new_hand:  # todo consider replacing with if self.new_hands_dealt():
            self.is_first_action = True
            self.hands_total += 1
            self.vpip_updated_this_hand = False
            self.pfr_updated_this_preflop = False
            self.tightness_updated_this_hand = False
            self.cbet_flop_updated_this_hand = False
            self.cbet_turn_updated_this_hand = False
            self.cbet_river_updated_this_hand = False
            self.three_bet_updated_this_hand = False
            hand_played = 1 if action > ActionSpace.FOLD and not self.big_blind_checked_preflop(obs, action) else 0
            self.hands_played += hand_played
        self._update_vpip(obs, action)
        self._update_af(obs, action)
        self._update_pfr(obs, action)
        self._update_cbet(obs, action)
        self._update_3bet(obs, action)
        self._update_tightness(obs, action)
        self.is_first_action = False
        if action == ActionSpace.FOLD:
            self.registered_folds += 1
        elif action == ActionSpace.CHECK_CALL:
            self.registered_calls += 1
        elif action >= ActionSpace.RAISE_MIN_OR_3BB:
            self.registered_raises += 1
        else:
            raise ValueError(f"Action must be in [0,1,2,3,4,5] but was {action}")

    def reset(self):
        """If necessary, we can reset all stats to 0 here, but I dont think well need it"""
        pass

    def to_dict(self):
        return {'name': self.pname,
                'vpip | showdown': 1,
                'vpip': self.vpip,
                'pfr': self.pfr,
                'tightness': self.tightness,
                '3bet': self.threebet,
                'cbet': self.cbet,
                'af': self.af,
                # 'hands_to_showdown': self.hands_to_showdown,  # compute wtsd
                'hands_played': self.hands_played,
                'hands_total': self.hands_total,
                'registered_folds': self.registered_folds,
                'registered_calls': self.registered_calls,
                'registered_raises': self.registered_raises,
                }

    def to_disk(self, fpath):
        # fpath: <ABS_PATH>.json
        d = self.to_dict()
        if not os.path.exists(Path(fpath).parent):
            os.makedirs(Path(fpath).parent)
        with open(fpath, "w+") as file:
            json_str = json.dumps(d, indent=4)
            file.write(json_str)
