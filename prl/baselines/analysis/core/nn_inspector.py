from typing import List, Tuple, Dict, Optional, Union, Any

import torch
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as fts
import numpy as np
from prl.environment.Wrappers.base import ActionSpace
from prl.environment.Wrappers.utils import init_wrapped_env
from prl.environment.steinberger.PokerRL.game.Poker import Poker
from prl.environment.steinberger.PokerRL.game.games import NoLimitHoldem

from prl.baselines.analysis.core.stats import PlayerStats
from prl.baselines.examples.examples_tianshou_env import make_default_tianshou_env
from prl.baselines.supervised_learning.data_acquisition.core.encoder import PlayerInfo, Positions6Max
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode, Action, Blind, \
    PlayerWithCards, ActionType
from prl.baselines.supervised_learning.data_acquisition.environment_utils import make_board_cards, card_tokens, card

MULTIPLY_BY = 100  # because env expects Integers, we convert $2,58 to $258


class Inspector:
    Observations = Optional[List[List]]
    Actions_Taken = Optional[List[Tuple[int, int]]]

    def __init__(self, baseline, env_wrapper_cls=None):
        self.env_wrapper_cls = env_wrapper_cls
        self._wrapped_env = None
        self._currency_symbol = None
        self._feature_names = None
        self.baseline = baseline  # inspection is computed against baseline
        self.logits_when_correct = {ActionSpace.FOLD: None,
                                    ActionSpace.CHECK_CALL: None,
                                    ActionSpace.RAISE_MIN_OR_3BB: None,
                                    ActionSpace.RAISE_6_BB: None,
                                    ActionSpace.RAISE_10_BB: None,
                                    ActionSpace.RAISE_20_BB: None,
                                    ActionSpace.RAISE_50_BB: None,
                                    ActionSpace.RAISE_ALL_IN: None,
                                    }  # for every wrong prediction: get all logits
        # self.true = {}  # for every true prediction: count prediction for each label
        self.label_counts_true = {ActionSpace.FOLD: torch.zeros(len(ActionSpace)),
                                  ActionSpace.CHECK_CALL: torch.zeros(len(ActionSpace)),
                                  ActionSpace.RAISE_MIN_OR_3BB: torch.zeros(len(ActionSpace)),
                                  ActionSpace.RAISE_6_BB: torch.zeros(len(ActionSpace)),
                                  ActionSpace.RAISE_10_BB: torch.zeros(len(ActionSpace)),
                                  ActionSpace.RAISE_20_BB: torch.zeros(len(ActionSpace)),
                                  ActionSpace.RAISE_50_BB: torch.zeros(len(ActionSpace)),
                                  ActionSpace.RAISE_ALL_IN: torch.zeros(len(ActionSpace))}
        self.logits_when_wrong = {ActionSpace.FOLD: None,
                                  ActionSpace.CHECK_CALL: None,
                                  ActionSpace.RAISE_MIN_OR_3BB: None,
                                  ActionSpace.RAISE_6_BB: None,
                                  ActionSpace.RAISE_10_BB: None,
                                  ActionSpace.RAISE_20_BB: None,
                                  ActionSpace.RAISE_50_BB: None,
                                  ActionSpace.RAISE_ALL_IN: None,
                                  }  # for every wrong prediction: get all logits
        # self.true = {}  # for every false prediction: count predictions for each label
        self.label_counts_false = {ActionSpace.FOLD: torch.zeros(len(ActionSpace)),
                                   ActionSpace.CHECK_CALL: torch.zeros(len(ActionSpace)),
                                   ActionSpace.RAISE_MIN_OR_3BB: torch.zeros(len(ActionSpace)),
                                   ActionSpace.RAISE_6_BB: torch.zeros(len(ActionSpace)),
                                   ActionSpace.RAISE_10_BB: torch.zeros(len(ActionSpace)),
                                   ActionSpace.RAISE_20_BB: torch.zeros(len(ActionSpace)),
                                   ActionSpace.RAISE_50_BB: torch.zeros(len(ActionSpace)),
                                   ActionSpace.RAISE_ALL_IN: torch.zeros(len(ActionSpace))}

    class _EnvironmentEdgeCaseEncounteredError(ValueError):
        """This error is thrown in rare cases where the PokerEnv written by Erich Steinberger,
        fails due to edge cases. I filtered these edge cases by hand, and labelled them with the hand id.
        """
        edge_case_one = {216163387520: """Player 3 (UTG) folds
                                Player 4 (MP) calls BB and is all-in
                                Player 5 (CU) folds
                                Player 0 (Button) folds
                                Player 1 (SB) folds
                                Player 2 (BB) - ENV is waiting for Player 2s action but the text file does not contain that action,
                                because it is implictly given:
                                BB can only check because the other player called the big blind and is all in anyway."""
                         }
        # fixed: edge_case_two = {213304492236: """Side Pots not split properly..."""}

    class _EnvironmentDidNotTerminateInTimeError(IndexError):
        """Edge cases we dont want to handle because we have enough training data already."""
        # '../../../data/0.25-0.50/BulkHands-14686/unzipped/PokerStars-NoLimitHoldem-0.25-0.50-6Max-Regular-20200505- 1 (0)/Parenago III-0.25-0.50-USD-NoLimitHoldem-PokerStars-5-5-2020.txt'
        # 213347137341: Player Folded when he could have checked

    @property
    def feature_names(self):
        return self._feature_names

    @staticmethod
    def str_cards_to_list(cards: str):
        """See example below """
        # '[6h Ts Td 9c Jc]'
        rm_brackets = cards.replace('[', '').replace(']', '')
        # '6h Ts Td 9c Jc'
        card_list = rm_brackets.split(' ')
        # ['6h', 'Ts', 'Td', '9c', 'Jc']
        return card_list

    @staticmethod
    def build_action(action: Action):
        """Be careful with float-to-int conversion:
        >>> round(8.62 * 100)
        862
        >>> int(8.62 * 100)
        861
        """
        return action.action_type.value, round(float(action.raise_amount) * MULTIPLY_BY)

    def make_blinds(self, blinds: List[Blind]):
        """Under Construction."""
        assert len(blinds) == 2
        sb = blinds[0]
        assert sb.type == 'small blind'
        bb = blinds[1]
        assert bb.type == 'big blind'
        return round(float(sb.amount.split(self._currency_symbol)[1]) * MULTIPLY_BY), \
            round(float(bb.amount.split(self._currency_symbol)[1]) * MULTIPLY_BY)

    def make_showdown_hands(self, table: Tuple[PlayerInfo], showdown: List[PlayerWithCards]):
        """Under Construction. """
        # initialize default hands
        default_card = [Poker.CARD_NOT_DEALT_TOKEN_1D, Poker.CARD_NOT_DEALT_TOKEN_1D]  # rank, suit
        player_hands = [[default_card, default_card] for _ in range(len(table))]

        # overwrite known hands
        for seat in table:
            for final_player in showdown:
                if seat.player_name == final_player.name:
                    # '[6h Ts]' to ['6h', 'Ts']
                    showdown_cards = card_tokens(final_player.cards)
                    # ['6h', 'Ts'] to [[5,3], [5,0]]
                    hand = [card(token) for token in showdown_cards]
                    # overwrite [[-127,127],[-127,-127]] with [[5,3], [5,0]]
                    player_hands[int(seat.position_index)] = hand
        return player_hands

    @staticmethod
    def _roll_position_indices(num_players: int, btn_idx: int) -> np.ndarray:
        """ Roll position indices, such that each seat is assigned correct position.
        Args:
          btn_idx: seat index (not seat number) of seat that is currently the Button.
                    Seats can be ["Seat 1", "Seat3", "Seat 5"]. If "Seat 5" is the Button,
                    then btn_idx=2
          num_players: Number of players currently at the table (not max. players).
        Returns: Assignment of position indices to seat numbers.

        Example: btn_idx=1
            # ==> np.roll([0,1,2], btn_idx) returns [2,0,1]:
            # The first  seat has position index 2, which is BB
            # The second seat has position index 0, which is BTN
            # The third  seat has position index 1, which is SB
        """
        # np.roll([0,1,2,3], 1) returns [3,0,1,2]  <== Button is at index 1 now
        return np.roll(np.arange(num_players), btn_idx)

    def make_table(self, episode: PokerEpisode) -> Tuple[PlayerInfo]:
        """Under Construction."""
        # Roll position indices, such that each seat is assigned correct position
        rolled_position_indices = self._roll_position_indices(episode.num_players, episode.btn_idx)

        # init {'BTN': None, 'SB': None,..., 'CO': None}
        player_info: Dict[str, PlayerInfo] = dict.fromkeys(
            [pos.name for pos in Positions6Max][:episode.num_players])

        # build PlayerInfo for each player
        for seat_idx, seat in enumerate(episode.player_stacks):
            seat_number = int(seat.seat_display_name[5])
            player_name = seat.player_name
            stack_size = round(float(seat.stack[1:]) * MULTIPLY_BY)
            position_index = rolled_position_indices[seat_idx]
            position = Positions6Max(position_index).name
            player_info[position] = PlayerInfo(seat_number,  # 2
                                               position_index,  # 0
                                               position,  # 'BTN'
                                               player_name,  # 'JoeSchmoe Billy'
                                               stack_size)

        # Seat indices such that button is first, regardless of seat number
        players_ordered_starting_with_button = [v for v in player_info.values()]
        return tuple(players_ordered_starting_with_button)

    def _build_cards_state_dict(self, table: Tuple[PlayerInfo], episode: PokerEpisode):
        """Under Construction."""
        board_cards = make_board_cards(episode.board_cards)
        # --- set deck ---
        # cards are drawn without ghost cards, so we simply replace the first 5 cards of the deck
        # with the board cards that we have parsed
        deck = np.empty(shape=(13 * 4, 2), dtype=np.int8)
        deck[:len(board_cards)] = board_cards
        # make hands: np.ndarray(shape=(n_players, 2, 2))
        player_hands = self.make_showdown_hands(table, episode.showdown_hands)
        initial_board = np.full((5, 2), Poker.CARD_NOT_DEALT_TOKEN_1D, dtype=np.int8)
        return {'deck': {'deck_remaining': deck},  # np.ndarray(shape=(52-n_cards*num_players, 2))
                'board': initial_board,  # np.ndarray(shape=(n_cards, 2))
                'hand': player_hands}

    def _make_ante(self, ante: str) -> float:
        """Converts ante string to float, e.g. '$0.00' -> float(0.00)"""
        return float(ante.split(self._currency_symbol)[1]) * MULTIPLY_BY

    def run_assertions(self, obs, i, done):
        hands = []
        for s in self._wrapped_env.env.seats:
            hands.append(s.hand)
        num_players = len(self._wrapped_env.env.seats)
        p0_first_card = obs[fts.First_player_card_0_rank_0:fts.First_player_card_1_rank_0]
        p0_second_card = obs[fts.First_player_card_1_rank_0:fts.Second_player_card_0_rank_0]
        p1_first_card = obs[fts.Second_player_card_0_rank_0:fts.Second_player_card_1_rank_0]
        p1_second_card = obs[fts.Second_player_card_1_rank_0:fts.Third_player_card_0_rank_0]
        r00 = hands[i][0][0]  # rank first player first card
        s00 = hands[i][0][1]
        r01 = hands[i][1][0]
        s01 = hands[i][1][1]  # suite first player second card
        """
        a0 calls after reset
        a1 observes obs1 folds
        a2 observes obs2 folds --> Game ended
        who gets obs3? a2 gets ob3 but a0 and a1 are also candidates. however they wont.
        for simplicity in these cases the transitions are cut out and only 
        the transition for a2 survives
        """
        # note after done we dont increment i, so the last remaining player gets obs
        if not sum(p0_first_card) == 0:
            # in case of specific initialization some agent cards might be unknown even to the env
            assert p0_first_card[r00] == 1
            assert p0_first_card[13 + s00] == 1
            assert p0_second_card[r01] == 1
            assert p0_second_card[13 + s01] == 1
        if not done:
            assert sum(p1_first_card) == 0
            assert sum(p1_second_card) == 0
        else:
            # make sure other palyer cards are seen. enough for next player as this will
            # trigger at some point. this function is called millions of times
            if hands[(i + 1) % num_players][0][0] != Poker.CARD_NOT_DEALT_TOKEN_1D:
                assert sum(p1_first_card) == 2
                assert sum(p1_second_card) == 2
    def _update_card(self):
        c = np.zeros(17)
        rand_rank = np.random.randint(0, 13)
        rand_suite = np.random.randint(13, 17)
        c[rand_rank] += 1
        c[rand_suite] += 1
        return c

    def update_card(self):
        obs_card_bits = self._update_card()
        # while we drew cards that already are on board or other player hand, repeat until we have available cards
        while np.any(np.all(obs_card_bits == self.occupied_cards, axis=1)):
            obs_card_bits = self._update_card()
        assert sum(obs_card_bits[:13] == 1)
        assert sum(obs_card_bits[13:] == 1)
        assert len(obs_card_bits) == 17
        return obs_card_bits

    def overwrite_hand_cards_with_random_cards(self, obs):
        """This is useful because if we only have showdown data,
        it will be skewed towards playable hands and there will be no data on weak hands."""
        # self.initial_villain_cards = obs[fts.First_player_card_0_rank_0:fts.First_player_card_1_rank_0],
        # obs[fts.First_player_card_1_rank_0:fts.First_player_card_1_suit_3+1]
        # remove old cards from observing player
        obs[fts.First_player_card_0_rank_0:fts.First_player_card_1_rank_0] = 0
        obs[fts.First_player_card_1_rank_0:fts.First_player_card_1_suit_3+1] = 0
        # todo sample new hand from remaining cards and overwrite observation
        # update c0
        obs_bits_c0 = self.update_card()
        obs[fts.First_player_card_0_rank_0:fts.First_player_card_1_rank_0] = obs_bits_c0
        # update c1
        obs_bits_c1 = self.update_card()
        obs[fts.First_player_card_1_rank_0:fts.First_player_card_1_suit_3+1] = obs_bits_c1
        assert sum(obs[fts.First_player_card_0_rank_0:fts.First_player_card_1_suit_3+1]) == 4

        return obs
    def _simulate_environment(self, pname, env, episode, cards_state_dict, table, starting_stack_sizes_list,
                              selected_players=None):
        """Under Construction."""
        # if episode.hand_id == 216163387520 or episode.hand_id == 214211025466:
        for s in starting_stack_sizes_list:
            if s == env.env.SMALL_BLIND or s == env.env.BIG_BLIND:
                raise self._EnvironmentEdgeCaseEncounteredError("Edge case 1 encountered. See docstring for details.")

        # todo step tianshou env as well and compare obs
        state_dict = {'deck_state_dict': cards_state_dict, 'table': table}
        obs, _, done, _ = env.reset(config=state_dict)
        obst_dict = self.tianshou_env.reset(options={'reset_config': state_dict})
        obs_tianshou = obst_dict['obs']
        legal_moves = obst_dict['mask']
        assert np.array_equal(obs, obs_tianshou)
        assert obs[-1] in [0, 1, 2, 3, 4, 5], f"obs[-1] = {obs[-1]}. " \
                                              f"get_current_obs should have caught this already. check the wrapper impl"

        # --- Step Environment with action --- #
        observations = []
        actions = []
        showdown_players: List[str] = [player.name for player in episode.showdown_hands]

        it = 0
        i = 0 if len(starting_stack_sizes_list) < 4 else 3
        debug_action_list = []
        while not done:
            try:
                action = episode.actions_total['as_sequence'][it]
            except IndexError:
                raise self._EnvironmentDidNotTerminateInTimeError
            action_formatted = self.build_action(action)
            next_to_act = env.current_player.seat_id
            for player in table:
                # if player reached showdown (we can see his cards)
                # can use showdown players actions and observations or use only top_players actions and observations
                filtered_players = showdown_players if not selected_players else [p for p in showdown_players if
                                                                                  p in selected_players]
                # filtered_players = [pname]
                # only store obs and action of acting player
                if player.position_index == next_to_act and player.player_name in filtered_players:
                    # uncomment this line to make random cards
                    obs = self.overwrite_hand_cards_with_random_cards(obs)
                    observations.append(obs)
                    action_label = self._wrapped_env.discretize(action_formatted)
                    # uncomment this line to set all actions to fold
                    action_label = ActionSpace.FOLD
                    actions.append(action_label)
                    pred = self.baseline.compute_action(obs, legal_moves)
                    # todo make one for winner and one for folds
                    # if not player.player_name in [winner.name for winner in episode.winners]:
                    #     # if not player.player_name in selected_players:
                    #     # replace action call/raise with fold
                    #     action_label = self._wrapped_env.discretize((ActionType.FOLD.value, -1))

                    # Update neural network stats

            debug_action_list.append(action_formatted)
            obs, _, done, _ = env.step(action_formatted)
            obs_dict, _, _, _, _ = self.tianshou_env.step(action_formatted)
            obs_tianshou = obs_dict['obs']
            assert np.array_equal(obs, obs_tianshou)
            it += 1
            if not done:
                i = self._wrapped_env.env.current_player.seat_id
            self.run_assertions(obs, i, done)

        if not observations:
            print(actions)
            raise RuntimeError("Seems we need more debugging")
        return observations, actions

    def cards2dtolist(self, cards2d):
        bits = np.zeros(17)
        bits[cards2d[0]] += 1
        bits[13 + cards2d[1]] += 1
        return bits
    def get_occupied_cards(self) -> List[np.ndarray]:
        bit_arr = []
        hands = self.state_dict['hand']
        board = self.state_dict['deck']['deck_remaining'][:5]  # before reset so all cards are in the deck in the order theyre drawn
        for hand in hands:
            if hand[0] != [-127, -127]:
                bit_arr.append(self.cards2dtolist(hand[0]))
                bit_arr.append(self.cards2dtolist(hand[1]))
        for card in board:
            bit_arr.append(self.cards2dtolist(card))
        return bit_arr
    def inspect_episode(self, episode: PokerEpisode, pname: str, selected_players=None) -> Union[
        tuple[None, None], tuple[list[Any], list[ActionSpace]]]:
        """Runs environment with steps from PokerEpisode.
        Returns observations and corresponding actions of players that made it to showdown."""

        # Maybe skip game, if selected_players is set and no selected player was in showdown
        if selected_players:
            # skip episode if no selected_players has played in it
            showdown_players: List[str] = [player.name for player in episode.showdown_hands]
            skip = True
            for s in selected_players:
                if s in showdown_players:
                    skip = False
                    break
            if skip:
                return None, None  # observations, actions are empty, if selected players were not part of showdown
        # utils
        table = self.make_table(episode)
        self._currency_symbol = episode.currency_symbol

        # Initialize environment for simulation of PokerEpisode
        # get starting stacks, starting with button at index 0
        stacks = [player.stack_size for player in table]
        self._wrapped_env = init_wrapped_env(self.env_wrapper_cls,
                                             stacks,
                                             multiply_by=1)
        sb, bb = self.make_blinds(episode.blinds)
        self._wrapped_env.env.SMALL_BLIND, self._wrapped_env.env.BIG_BLIND = sb, bb
        self._wrapped_env.env.ANTE = self._make_ante(episode.ante)
        cards_state_dict = self._build_cards_state_dict(table, episode)
        self.state_dict = cards_state_dict
        self.occupied_cards = self.get_occupied_cards()
        agent_names = np.roll([a.player_name for a in episode.player_stacks], -episode.btn_idx)
        self.tianshou_env = make_default_tianshou_env(stack_sizes=stacks,
                                                      blinds=[sb, bb],
                                                      agents=agent_names,
                                                      num_players=len(agent_names))
        # Collect observations and actions, observations are possibly augmented
        try:
            return self._simulate_environment(env=self._wrapped_env,
                                              episode=episode,
                                              cards_state_dict=cards_state_dict,
                                              table=table,
                                              starting_stack_sizes_list=stacks,
                                              selected_players=selected_players,
                                              pname=pname)
        except self._EnvironmentEdgeCaseEncounteredError:
            return None, None
        except self._EnvironmentDidNotTerminateInTimeError:
            return None, None
