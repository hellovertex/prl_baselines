from typing import List, Tuple, Optional

import numpy as np
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as fts, AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env
from prl.environment.steinberger.PokerRL.game.Poker import Poker

from prl.baselines.supervised_learning.data_acquisition.core.encoder import PlayerInfo, Positions6Max
from prl.baselines.supervised_learning.data_acquisition.core.parser import Action, Blind, \
    PlayerWithCards
from prl.baselines.supervised_learning.data_acquisition.environment_utils import card_tokens, card, make_board_cards
from prl.baselines.supervised_learning.v2.poker_model import PokerEpisodeV2, Player

MULTIPLY_BY = 100  # because env expects Integers, we convert $2,58 to $258


class EncoderV2:
    Observations = Optional[List[List]]
    Actions_Taken = Optional[List[Tuple[int, int]]]

    def __init__(self, verbose=False):
        # self.no_folds = no_folds
        self.verbose = verbose
        self.env_wrapper_cls = AugmentObservationWrapper
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
        return sb.amount, bb.amount

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

    def _build_cards_state_dict(self, episode: PokerEpisodeV2):
        """Under Construction."""
        n_players = len(episode.players)
        # board_cards = make_board_cards(episode.board_cards)
        # --- set deck ---
        # cards are drawn without ghost cards, so we simply replace the first 5 cards of the deck
        # with the board cards that we have parsed
        deck = np.empty(shape=(13 * 4, 2), dtype=np.int8)
        # deck[:len(board_cards)] = board_cards
        # make hands: np.ndarray(shape=(n_players, 2, 2))
        # player_hands = self.make_showdown_hands(table, episode.showdown_hands)
        initial_board = np.full((5, 2), Poker.CARD_NOT_DEALT_TOKEN_1D, dtype=np.int8)
        return {'deck': {'deck_remaining': deck},  # np.ndarray(shape=(52-n_cards*num_players, 2))
                'board': initial_board,  # np.ndarray(shape=(n_cards, 2))
                'hand': np.zeros((n_players, 2, 2))}

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
        obs[fts.First_player_card_1_rank_0:fts.First_player_card_1_suit_3 + 1] = 0
        # todo sample new hand from remaining cards and overwrite observation
        # update c0
        obs_bits_c0 = self.update_card()
        obs[fts.First_player_card_0_rank_0:fts.First_player_card_1_rank_0] = obs_bits_c0
        # update c1
        obs_bits_c1 = self.update_card()
        obs[fts.First_player_card_1_rank_0:fts.First_player_card_1_suit_3 + 1] = obs_bits_c1
        assert sum(obs[fts.First_player_card_0_rank_0:fts.First_player_card_1_suit_3 + 1]) == 4

        return obs

    def _simulate_environment(self,
                              env,
                              episode: PokerEpisodeV2,
                              cards_state_dict,
                              starting_stack_sizes_list,
                              selected_players=None):
        """Runs poker environment from episode and returns observations and actions emitted for selected players.
        If no players selectd, then all showdown players actions and observations will be returned."""
        # if episode.hand_id == 216163387520 or episode.hand_id == 214211025466:
        for s in starting_stack_sizes_list:
            if s == env.env.SMALL_BLIND or s == env.env.BIG_BLIND:
                # skip edge case of player all in by calling big blind
                raise self._EnvironmentEdgeCaseEncounteredError("Edge case 1 encountered. See docstring for details.")
        state_dict = {'deck_state_dict': cards_state_dict}
        if episode.hand_id == 220485600493:
            print('break at this line for debug')
            a = 1
        obs, _, done, _ = env.reset(config=state_dict)
        assert obs[-1] in [0, 1, 2, 3, 4, 5], f"obs[-1] = {obs[-1]}. " \
                                              f"get_current_obs should have caught this already. check the wrapper impl"
        # --- Step Environment with action --- #
        observations = []
        actions = []
        showdown_players: List[str] = [player.name for player in episode.showdown_hands]
        # if player reached showdown we can see his cards
        # filtered_players = showdown_players if not selected_players else [p for p in showdown_players if
        #                                                                           p in selected_players]
        filtered_players = None
        if not selected_players:
            filtered_players = showdown_players
        else:
            for p in showdown_players:
                if p in selected_players:
                    filtered_players = showdown_players
        assert filtered_players is not None, "filtered players must be equal to showdown players"
        it = 0
        debug_action_list = []
        while not done:
            try:
                action = episode.actions_total['as_sequence'][it]
            except IndexError:
                raise self._EnvironmentDidNotTerminateInTimeError

            action_formatted = self.build_action(action)
            action_label = self._wrapped_env.discretize(action_formatted)
            next_to_act = env.current_player.seat_id
            # if self.verbose:
            #     pretty_print(next_to_act, obs, action_label)
            for player in episode.players:
                pass
                # If player is showdown player
                # if player.name == action.who and player.name in selected_players:
                # if player.position_index == next_to_act and player.player_name in filtered_players:
                #     # take (obs, action) from selected player or winner if selected players is None
                #     target_players = selected_players if selected_players else [winner.name for winner in
                #                                                                 episode.winners]
                #     if player.name in target_players:
                #         actions.append(action_label)
                #         observations.append(obs)
                #     # Maybe select opponents (obs, action) where we set action=FOLD
                #     # Note that opponent may be the winner if selected players is not None,
                #     # but we assume that selected players have the better strategy even
                #     # if they lose to variance here, so we drop the winners action in this case
                #     else:
                #         if not self.drop_folds:
                #             action_label = self._wrapped_env.discretize((ActionType.FOLD.value, -1))
                #             if self.randomize_fold_cards:
                #                 obs = self.overwrite_hand_cards_with_random_cards(obs)
                #             actions.append(action_label)
                #             observations.append(obs)
            debug_action_list.append(action_formatted)

            obs, _, done, _ = env.step(action_formatted)
            it += 1

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
        board = self.state_dict['deck']['deck_remaining'][
                :5]  # before reset so all cards are in the deck in the order theyre drawn
        for hand in hands:
            if hand[0] != [-127, -127]:
                bit_arr.append(self.cards2dtolist(hand[0]))
                bit_arr.append(self.cards2dtolist(hand[1]))
        for card in board:
            bit_arr.append(self.cards2dtolist(card))
        return bit_arr

    def get_players_starting_with_button(self, episode: PokerEpisodeV2) -> List[Player]:
        has_moved = {}
        num_players = len(episode.players)
        for pname, _ in episode.players.items():
            has_moved[pname] = False
        players_sorted = []
        for action in episode.actions['as_sequence']:
            if has_moved[action.who]:
                break
            has_moved[action.who] = True
            players_sorted.append(episode.players[action.who])
        if len(players_sorted) == len(episode.players) -1:
            # big blind collected uncalled bet, so he/she did not perform any action
            for pname, pinfo in episode.players.items():
                if pinfo.position == Positions6Max.BB:
                    players_sorted.append(pinfo)
        assert len(players_sorted) == len(episode.players)
        # [3,4,0,1,2] ->
        if num_players > 3:
            players_sorted = np.roll(players_sorted, -(num_players-3))
        assert players_sorted[0].seat_num_one_indexed == episode.btn_seat_num_one_indexed
        return players_sorted

    def make_player_hands(self, episode):

        for pname, pinfo in episode.players.items():
            if pinfo.cards:
                # In: '[Qs Qd]'
                # Out: [[10, 1], [10, 2]]
                pass
                a = 1
        return []

    def encode_episode(self,
                       episode: PokerEpisodeV2,
                       drop_folds,
                       randomize_fold_cards,
                       selected_players,
                       verbose) -> Tuple[
        Observations, Actions_Taken]:
        """Runs environment with steps from PokerEpisode.
        Returns observations and corresponding actions of players that made it to showdown."""
        # todo: for each selected player
        #  pick (obs, action)
        #  if players cards are unknown (no showdown) randomize them
        #  if action is fold -- terminate and return all observations
        self.verbose = verbose
        self.drop_folds = drop_folds
        self.randomize_fold_cards = randomize_fold_cards
        self._currency_symbol = episode.currency_symbol
        players = self.get_players_starting_with_button(episode)
        stacks = [player.stack for player in players]
        sb, bb = episode.blinds['sb'], episode.blinds['bb']
        self._wrapped_env = init_wrapped_env(self.env_wrapper_cls,
                                             stacks,
                                             blinds=(sb, bb),
                                             multiply_by=1,  # already multiplied in self.make_table()
                                             )
        # will be used for naming feature index in training data vector
        self._feature_names = list(self._wrapped_env.obs_idx_dict.keys())

        self._wrapped_env.env.SMALL_BLIND = sb
        self._wrapped_env.env.BIG_BLIND = bb
        self._wrapped_env.env.ANTE = 0.0
        cards_state_dict = self._build_cards_state_dict(episode)
        state_dict = {}
        board = make_board_cards(episode.board)
        player_hands = self.make_player_hands(episode)
        self.state_dict = {}
        # self.occupied_cards = self.get_occupied_cards()
        # # Collect observations and actions, observations are possibly augmented
        # try:
        #     return self._simulate_environment(env=self._wrapped_env,
        #                                       episode=episode,
        #                                       cards_state_dict=cards_state_dict,
        #                                       starting_stack_sizes_list=stacks,
        #                                       selected_players=selected_players)
        # except self._EnvironmentEdgeCaseEncounteredError:
        #     return None, None
        # except self._EnvironmentDidNotTerminateInTimeError:
        #     return None, None
# in: .txt files
# out: .csv files? maybe npz or easier-on-memory formats preferred?
