from typing import List, Tuple, Dict, Optional

import numpy as np
from prl.environment.steinberger.PokerRL.game.Poker import Poker
from prl.environment.steinberger.PokerRL.game.games import NoLimitHoldem

from core.encoder import Encoder, PlayerInfo, Positions6Max
from core.parser import PokerEpisode, Action, ActionType, Blind

DICT_RANK = {'': -127,
             '2': 0,
             '3': 1,
             '4': 2,
             '5': 3,
             '6': 4,
             '7': 5,
             '8': 6,
             '9': 7,
             'T': 8,
             'J': 9,
             'Q': 10,
             'K': 11,
             'A': 12}

DICT_SUITE = {'': -127,
              'h': 0,
              'd': 1,
              's': 2,
              'c': 3}


class RLStateEncoder(Encoder):
    Observations = Optional[List[List]]
    Actions_Taken = Optional[List[Tuple[int, int]]]

    def __init__(self, env_wrapper_cls=None):
        self.env_wrapper_cls = env_wrapper_cls
        self._wrapped_env = None
        self._currency_symbol = None
        self._feature_names = None

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
    def build_action(action: Action, multiply_by=100):
        """Under Construction."""
        return action.action_type.value, round(float(action.raise_amount) * multiply_by)

    def make_blinds(self, blinds: List[Blind], multiply_by: int = 100):
        """Under Construction."""
        assert len(blinds) == 2
        sb = blinds[0]
        assert sb.type == 'small blind'
        bb = blinds[1]
        assert bb.type == 'big blind'
        return int(float(sb.amount.split(self._currency_symbol)[1]) * multiply_by), \
               int(float(bb.amount.split(self._currency_symbol)[1]) * multiply_by)

    def make_board_cards(self, board_cards: str):
        """Return 5 cards that we can prepend to the card deck so that the board will be drawn.
      Args:
        board_cards: for example '[6h Ts Td 9c Jc]'
      Returns:
        representation of board_cards that is understood by rl_env
        Example:
    """
        # '[6h Ts Td 9c Jc]' to ['6h', 'Ts', 'Td', '9c', 'Jc']
        card_list = self.str_cards_to_list(board_cards)
        assert len(card_list) == 5

        return [[DICT_RANK[card[0]], DICT_SUITE[card[1]]] for card in card_list]

    def make_showdown_hands(self, table, showdown):
        """Under Construction. """
        # initialize default hands
        default_card = [Poker.CARD_NOT_DEALT_TOKEN_1D, Poker.CARD_NOT_DEALT_TOKEN_1D]  # rank, suit
        player_hands = [[default_card, default_card] for _ in range(len(table))]

        # overwrite known hands
        for seat in table:
            for final_player in showdown:
                if seat.player_name == final_player.name:
                    # '[6h Ts]' to ['6h', 'Ts']
                    showdown_cards = self.str_cards_to_list(final_player.cards)
                    # ['6h', 'Ts'] to [[5,3], [5,0]]
                    hand = [[DICT_RANK[card[0]], DICT_SUITE[card[1]]] for card in showdown_cards]
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
            stack_size = float(seat.stack[1:])
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
        board_cards = self.make_board_cards(episode.board_cards)
        # --- set deck ---
        # cards are drawn without ghost cards, so we simply replace the first 5 cards of the deck
        # with the board cards that we have parsed
        deck = np.empty(shape=(13 * 4, 2), dtype=np.int8)
        deck[:len(board_cards)] = board_cards
        # make hands: np.ndarray(shape=(n_players, 2, 2))
        player_hands = self.make_showdown_hands(table, episode.showdown_hands)
        board = np.full((5, 2), Poker.CARD_NOT_DEALT_TOKEN_1D, dtype=np.int8)
        return {'deck': {'deck_remaining': deck},  # np.ndarray(shape=(52-n_cards*num_players, 2))
                'board': board,  # np.ndarray(shape=(n_cards, 2))
                'hand': player_hands}

    def _init_wrapped_env(self, table: Tuple[PlayerInfo], ante, multiply_by=100):
        """Initializes environment used to generate observations.
        Assumes Btn is at index 0."""
        # get starting stacks, starting with button at index 0
        stacks = [player.stack_size for player in table]
        starting_stack_sizes_list = [int(float(stack) * multiply_by) for stack in stacks]

        # make args for env
        args = NoLimitHoldem.ARGS_CLS(n_seats=len(table),
                                      starting_stack_sizes_list=starting_stack_sizes_list)
        # return wrapped env instance
        env = NoLimitHoldem(is_evaluating=True,
                            env_args=args,
                            lut_holder=NoLimitHoldem.get_lut_holder())
        self._wrapped_env = self.env_wrapper_cls(env)
        # will be used for naming feature index in training data vector
        self._feature_names = list(self._wrapped_env.obs_idx_dict.keys()) + ["button_index"]
        return starting_stack_sizes_list

    def _make_ante(self, ante: str, multiply_by=100) -> float:
        """Converts ante string to float, e.g. '$0.00' -> float(0.00)"""
        return float(ante.split(self._currency_symbol)[1]) * multiply_by

    def _simulate_environment(self, env, episode, cards_state_dict, table, starting_stack_sizes_list,
                              selected_players=None):
        """Under Construction."""
        # if episode.hand_id == 216163387520 or episode.hand_id == 214211025466:
        for s in starting_stack_sizes_list:
            if s == env.env.SMALL_BLIND or s == env.env.BIG_BLIND:
                # skip edge case of player all in by calling big blind
                raise self._EnvironmentEdgeCaseEncounteredError("Edge case 1 encountered. See docstring for details.")
        #
        # if episode.hand_id == 213304492236:
        #    debug = 1

        state_dict = {'deck_state_dict': cards_state_dict, 'table': table}
        obs, _, done, _ = env.reset(config=state_dict)
        assert obs[-1] in [0, 1, 2, 3, 4, 5], f"obs[-1] = {obs[-1]}. " \
                                              f"get_current_obs should have caught this already. check the wrapper impl"
        # --- Step Environment with action --- #
        observations = []
        actions = []
        showdown_players: List[str] = [player.name for player in episode.showdown_hands]
        skip = False
        # Maybe skip game, if selected_players is set and no selected player was in showdown
        if selected_players:
            skip = True
            for s in selected_players:
                if s in showdown_players:
                    skip = False
                    break
        if skip:
            return None, None  # observations, actions are empty, if selected players were not part of showdown

        it = 0
        debug_action_list = []
        while not done:
            try:
                action = episode.actions_total['as_sequence'][it]
                # # use only observations, for which next_action.player_name == top_player
                # if it+1 < len(episode.actions_total['as_sequence']):
                #     next_action = episode.actions_total['as_sequence'][it+1]

            except IndexError:
                raise self._EnvironmentDidNotTerminateInTimeError

            action_formatted = self.build_action(action)
            # store up to two actions per player per stage
            # self._actions_per_stage[action.player_name][action.stage].append(action_formatted)
            next_to_act = env.current_player.seat_id
            for player in table:
                # if player reached showdown (we can see his cards)
                # can use showdown players actions and observations or use only top_players actions and observations
                filtered_players = showdown_players if not selected_players else [p for p in showdown_players if
                                                                                  p in selected_players]
                # only store obs and action of acting player
                if player.position_index == next_to_act and player.player_name in filtered_players:
                    observations.append(obs)
                    # use showdown player actions as labels, 0 for loser and action for winner
                    if not selected_players:
                        # player that won showdown -- can be multiple (split pot)
                        if player.player_name in [winner.name for winner in episode.winners]:
                            action_label = self._wrapped_env.discretize(action_formatted)
                            # actions.append(action_formatted)  # use his action as supervised label
                        # player that lost showdown
                        else:
                            # replace action call/raise with fold
                            action_label = self._wrapped_env.discretize((ActionType.FOLD.value, -1))
                            # actions.append((ActionType.FOLD.value, -1))  # replace action with FOLD for now
                    # use top players actions as labels, take actions as labels directly 0 for fold 1 for checkcall etc
                    else:
                        action_label = self._wrapped_env.discretize(action_formatted)

                    actions.append(action_label)
            debug_action_list.append(action_formatted)
            obs, _, done, _ = env.step(action_formatted)
            it += 1

        if not observations:
            print(actions)
            raise RuntimeError("Seems we need more debugging")
        return observations, actions

    def encode_episode(self, episode: PokerEpisode, selected_players=None) -> Tuple[Observations, Actions_Taken]:
        """Runs environment with steps from PokerEpisode.
        Returns observations and corresponding actions of players that made it to showdown."""
        # skip episode if no selected_players has played in it
        if selected_players:
            skip = True
            for p_info in episode.player_stacks:
                if p_info.player_name in selected_players:
                    skip = False
                    break
            if skip:
                return None, None
        # utils
        table = self.make_table(episode)
        self._currency_symbol = episode.currency_symbol

        # Initialize environment for simulation of PokerEpisode
        # todo: pass env_cls as argument (N_BOARD_CARDS etc. gets accessible)
        starting_stack_sizes_list = self._init_wrapped_env(table, ante=episode.ante)

        self._wrapped_env.env.SMALL_BLIND, self._wrapped_env.env.BIG_BLIND = self.make_blinds(episode.blinds,
                                                                                              multiply_by=100)
        self._wrapped_env.env.ANTE = self._make_ante(episode.ante)
        cards_state_dict = self._build_cards_state_dict(table, episode)

        # Collect observations and actions, observations are possibly augmented
        try:
            return self._simulate_environment(env=self._wrapped_env,
                                              episode=episode,
                                              cards_state_dict=cards_state_dict,
                                              table=table,
                                              starting_stack_sizes_list=starting_stack_sizes_list,
                                              selected_players=selected_players)
        except self._EnvironmentEdgeCaseEncounteredError:
            return None, None
        except self._EnvironmentDidNotTerminateInTimeError:
            return None, None
