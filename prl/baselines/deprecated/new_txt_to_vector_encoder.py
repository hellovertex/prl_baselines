import random
import time
from typing import List, Tuple, Optional

import numpy as np
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as fts, AugmentObservationWrapper
from prl.environment.Wrappers.base import ActionSpace
from prl.environment.Wrappers.utils import init_wrapped_env
from prl.environment.steinberger.PokerRL import NoLimitHoldem
from prl.environment.steinberger.PokerRL.game.Poker import Poker

from prl.baselines.evaluation.utils import get_player_cards, get_board_cards, get_round
from prl.baselines.supervised_learning.data_acquisition.core.encoder import Positions6Max
from prl.baselines.supervised_learning.data_acquisition.core.parser import Blind
from prl.baselines.supervised_learning.data_acquisition.environment_utils import card_tokens, card, make_board_cards
from prl.baselines.supervised_learning.v2.poker_model import PokerEpisodeV2, Player, Action

MULTIPLY_BY = 100  # because env expects Integers, we convert $2,58 to $258
INVISIBLE_CARD = [Poker.CARD_NOT_DEALT_TOKEN_1D, Poker.CARD_NOT_DEALT_TOKEN_1D]
DEFAULT_HAND = [INVISIBLE_CARD, INVISIBLE_CARD]


class EncoderV2:
    Observations = Optional[List[List]]
    Actions_Taken = Optional[List[Tuple[int, int]]]

    def __init__(self, env, verbose=False):
        self.env = env
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
        return action.what, action.how_much

    def make_blinds(self, blinds: List[Blind]):
        """Under Construction."""
        assert len(blinds) == 2
        sb = blinds[0]
        assert sb.type == 'small blind'
        bb = blinds[1]
        assert bb.type == 'big blind'
        return sb.amount, bb.amount

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
                              episode,
                              players: List[Player],
                              action_list: List[Action],
                              selected_players=None):
        """Runs poker environment from episode and returns observations and actions emitted for selected players.
        If no players selectd, then all showdown players actions and observations will be returned."""
        # if episode.hand_id == 216163387520 or episode.hand_id == 214211025466:

        state_dict = {'deck_state_dict': self.state_dict}
        obs, _, done, _ = self.env.reset(config=state_dict)
        # todo add tianshou env and assert np.array_equal(obs, obs_tianshou)
        assert obs[-1] in [0, 1, 2, 3, 4, 5], f"obs[-1] = {obs[-1]}. " \
                                              f"get_current_obs should have caught this already. check the wrapper impl"
        # --- Step Environment with action --- #
        observations = []
        actions = []
        it = 0
        debug_action_list = []
        remaining_selected_players = []

        for player in players:
            cond = player in episode.winners if self.only_winners else True
            if player.name in selected_players and cond:
                remaining_selected_players.append(player.name)

        while not done:
            try:
                action = action_list[it]
            except IndexError:
                raise self._EnvironmentDidNotTerminateInTimeError

            action_formatted = self.build_action(action)
            action_label = self.env.discretize(action_formatted)
            next_to_act = action.who
            for player in players:
                if player.name == next_to_act:
                    # cond = player.is_showdown_player if self.drop_folds else True
                    cond = player in episode.winners if self.drop_folds else True
                    if player.name in remaining_selected_players and cond:
                        observations.append(obs)
                        actions.append(action_label)
                        cards = get_player_cards(obs)[0]
                        board = get_board_cards(obs)
                        boardcards = ""
                        for card in board:
                            boardcards += card
                        round = get_round(obs).lower()
                        assert action.stage == round, f'Failed with {action} and {round} for hand id {episode.hand_id}'
                        for card in boardcards:
                            assert card in episode.board, f'Failed with {card} and {episode.board}'
                        if player.cards:
                            assert cards.replace(',','') == player.cards

                        if action_label == ActionSpace.FOLD:
                            remaining_selected_players.remove(player.name)
                    elif player.name in episode.showdown_players and not self.drop_folds:
                        observations.append(obs)
                        actions.append(ActionSpace.FOLD.value)

            debug_action_list.append(action_formatted)
            if not remaining_selected_players:
                return observations, actions
            obs, _, done, _ = self.env.step(action_formatted)
            it += 1

        if not observations:
            assert len(remaining_selected_players) == 1
            pname = remaining_selected_players[0]
            assert episode.players[pname].position == Positions6Max.BB
            # big blind returned to player because every body folded so he/she didnt get to act
            return [], []
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
            if not np.array_equal(hand[0], INVISIBLE_CARD):
                bit_arr.append(self.cards2dtolist(hand[0]))
                bit_arr.append(self.cards2dtolist(hand[1]))
        for card in board:
            if not np.array_equal(card, INVISIBLE_CARD):
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
        if len(players_sorted) == len(episode.players) - 1:
            # big blind collected uncalled bet, so he/she did not perform any action
            for pname, pinfo in episode.players.items():
                if pinfo.position == Positions6Max.BB:
                    players_sorted.append(pinfo)
        assert len(players_sorted) == len(episode.players), f'Failed for episode {episode.hand_id}'
        # [3,4,0,1,2] ->
        if num_players > 3:
            players_sorted = np.roll(players_sorted, -(num_players - 3))
        assert players_sorted[0].seat_num_one_indexed == episode.btn_seat_num_one_indexed, f'Failed for episode {episode.hand_id}'
        return players_sorted

    def remove_cards(self, deck, removed_cards):
        deck = deck.tolist()
        new_deck = []
        for c in deck:
            if not c in removed_cards:
                new_deck.append(c)
        return new_deck

    def replace(self, cards, to_overwrite, replace_with):
        cards = np.array(cards)
        to_overwrite = np.array(to_overwrite)
        replace_with = np.array(replace_with)
        mask = np.isin(cards, to_overwrite, axis=1).all(axis=1)
        cards[mask] = replace_with
        return cards.tolist()

    def make_player_hands(self, players: List[Player], board: List):
        hands = []

        occupied_cards = board
        for pinfo in players:
            if pinfo.cards:
                # In: '[Qs Qd]' Out: [[10,2],[10,3]]
                cards = card_tokens(pinfo.cards)
                hand = [card(token) for token in cards]
                # hands.append(hand)
                occupied_cards.append(hand[0])
                occupied_cards.append(hand[1])

        deck = np.array([[rank, suit] for rank in range(13) for suit in range(4)])
        deck = self.remove_cards(deck, occupied_cards)
        for pinfo in players:
            if pinfo.cards:
                # In: '[Qs Qd]' Out: [[10,2],[10,3]]
                cards = card_tokens(pinfo.cards)
                hand = [card(token) for token in cards]
                hands.append(hand)
            else:
                # todo: implement this conditional: if self.fold_random_cards:
                # overwrite default hands with random cards that are not board or player cards
                idx0 = random.randint(0, len(deck) - 1)
                c0 = deck.pop(idx0)
                random_hand = [c0]
                idx1 = random.randint(0, len(deck) - 1)
                c1 = deck.pop(idx1)
                random_hand.append(c1)
                hands.append(random_hand)
        return hands

    def encode_episode(self,
                       episode: PokerEpisodeV2,
                       drop_folds: bool,
                       only_winners: False,
                       randomize_fold_cards: bool,
                       selected_players: List[str],
                       limit_num_players: int = None,
                       verbose: bool=False) -> Tuple[
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
        self.only_winners = only_winners
        if drop_folds:
            skip_hand = True
            target_players = episode.winners if only_winners else episode.showdown_players
            for p in target_players:
                if p.name in selected_players:
                    if p.cards:
                        skip_hand = False
            if skip_hand:
                return None, None

        players = self.get_players_starting_with_button(episode)
        if limit_num_players:
            if len(players) < limit_num_players:
                return None,None
        stacks = [player.stack for player in players]
        sb, bb = episode.blinds['sb'], episode.blinds['bb']
        # self._wrapped_env = init_wrapped_env(self.env_wrapper_cls,
        #                                      stacks,
        #                                      blinds=(sb, bb),
        #                                      multiply_by=1,  # already multiplied in self.make_table()
        #                                      )
        args = NoLimitHoldem.ARGS_CLS(n_seats=len(stacks),
                                      scale_rewards=False,
                                      use_simplified_headsup_obs=False,
                                      starting_stack_sizes_list=stacks)
        self.env.overwrite_args(args)
        # will be used for naming feature index in training data vector
        self._feature_names = list(self.env.obs_idx_dict.keys())
        self.env.env.SMALL_BLIND = sb
        self.env.env.BIG_BLIND = bb
        self.env.env.ANTE = 0.0
        deck = np.full(shape=(13 * 4, 2), fill_value=Poker.CARD_NOT_DEALT_TOKEN_1D, dtype=np.int8)
        board = make_board_cards(episode.board)
        if board:
            deck[:len(board)] = board
        else:
            assert not episode.actions['actions_flop']
            assert not episode.actions['actions_turn']
            assert not episode.actions['actions_river']
        player_hands = self.make_player_hands(players, board)
        initial_board = np.full((5, 2), Poker.CARD_NOT_DEALT_TOKEN_1D, dtype=np.int8)
        self.state_dict = {'deck': {'deck_remaining': deck},
                           'board': initial_board,
                           'hand': player_hands}
        self.occupied_cards = self.get_occupied_cards()

        for s in stacks:
            if s == self.env.env.SMALL_BLIND or s == self.env.env.BIG_BLIND:
                # skip edge case of player all in by calling big blind
                return None, None

        # Collect observations and actions, observations are possibly augmented

        try:
            # t0 = time.time()
            res = self._simulate_environment(episode,
                                             players,
                                             episode.actions['as_sequence'],
                                             selected_players=selected_players)
            # print(f'Simulation took {time.time() - t0} seconds')
            return res
        except self._EnvironmentEdgeCaseEncounteredError:
            return None, None
        except self._EnvironmentDidNotTerminateInTimeError:
            return None, None
        except AssertionError as e:
            print(e)
            return None, None
# in: .txt files
# out: .csv files? maybe npz or easier-on-memory formats preferred?
