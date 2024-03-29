import logging
import random
from typing import List, Tuple, Optional

import numpy as np
from prl.environment.Wrappers.aoh import Positions6Max as pos
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as fts, \
    AugmentObservationWrapper
from prl.environment.Wrappers.augment import FeaturesWithHudStats
from prl.environment.Wrappers.base import ActionSpace
from prl.environment.steinberger.PokerRL import NoLimitHoldem
from prl.environment.steinberger.PokerRL.game.Poker import Poker

from prl.baselines.cpp_hand_evaluator.monte_carlo import HandEvaluator_MonteCarlo
from prl.baselines.supervised_learning.data_acquisition.core.encoder import Positions6Max
from prl.baselines.supervised_learning.data_acquisition.core.parser import Blind
from prl.baselines.supervised_learning.data_acquisition.environment_utils import \
    card_tokens, card, make_board_cards
from prl.baselines.supervised_learning.v2.datasets.dataset_config import ActionGenOption
from prl.baselines.supervised_learning.v2.poker_model import PokerEpisodeV2, Player, \
    Action

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
        self.turn_ordered_positions = {2: (pos.BTN, pos.BB),
                                       3: (pos.BTN, pos.SB, pos.BB),
                                       4: (pos.CO, pos.BTN, pos.SB, pos.BB),
                                       5: (pos.MP, pos.CO, pos.BTN, pos.SB, pos.BB),
                                       6: (
                                           pos.UTG, pos.MP, pos.CO, pos.BTN, pos.SB,
                                           pos.BB)}
        self.positions = None
        self.mc_simulator = HandEvaluator_MonteCarlo()
        self._lut = None
        # self._feature_names = None
        # positions_from_btn = {2: (pos.BTN, pos.BB),
        #                       3: (pos.BTN, pos.SB, pos.BB),
        #                       4: (pos.BTN, pos.SB, pos.BB, pos.CO),
        #                       5: (pos.BTN, pos.SB, pos.BB, pos.MP, pos.CO),
        #                       6: (pos.BTN, pos.SB, pos.BB, pos.UTG, pos.MP, pos.CO)}
        # turn_ordered_positions = {2: (pos.BTN, pos.BB),
        #                           3: (pos.BTN, pos.SB, pos.BB),
        #                           4: (pos.CO, pos.BTN, pos.SB, pos.BB),
        #                           5: (pos.MP, pos.CO, pos.BTN, pos.SB, pos.BB),
        #                           6: (pos.UTG, pos.MP, pos.CO, pos.BTN, pos.SB, pos.BB)}
        # self.positions_from_btn = positions_from_btn[num_players]
        # self.turn_ordered_positions = turn_ordered_positions[num_players]

    @property
    def lut(self):
        return self._lut

    @lut.setter
    def lut(self, value):
        self._lut = value

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
        assert len(blinds) == 2, 'Only Small blind and big blind allowed'
        sb = blinds[0]
        assert sb.type == 'small blind', 'sb.type must equal `small blind`'
        bb = blinds[1]
        assert bb.type == 'big blind'
        return sb.amount, bb.amount, 'bb.type must equal `big blind`'

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
        assert sum(obs_card_bits[:13] == 1), 'Card bits must one hot encode suite'
        assert sum(obs_card_bits[13:] == 1), 'Card bits must one hot encode rank'
        assert len(obs_card_bits) == 17, 'Card bits must have length 17'
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
        obs[
        fts.First_player_card_1_rank_0:fts.First_player_card_1_suit_3 + 1] = obs_bits_c1
        assert sum(
            obs[fts.First_player_card_0_rank_0:fts.First_player_card_1_suit_3 + 1]) == 4, \
            'sum of card bits must be 2+2=4'

        return obs

    def append_hud_stats(self,
                         obs,
                         hero: Player,
                         players,
                         n_opponents,
                         n_iter=5000):
        """
        'Player_0_is_tight',  # set only if n_games > 50
        'Player_0_is_aggressive',  # set only if n_games > 50
        """
        # get hand_ids for each player
        # then when computing vpip af pfr, use only hand_ids['player_name']
        hero_name = hero.name

        hud = np.zeros(18)
        # In case the player did fold -- we do not have to run Monte Carlo
        # and simply set win_prob to 0.2, not 0 because we dont want to encode fold in obs
        if hero.cards is None:
            win_prob = 0.2
        else:
            mc_results = self.mc_simulator.run_mc(obs,
                                                  n_opponents=n_opponents,
                                                  n_iter=n_iter)
            win_prob = (mc_results['won'] + mc_results['tied']) / n_iter
        players = np.roll(players, -players.index(hero_name))
        for offset, opponent in enumerate(players[1:]):
            # perform lookup
            d = self.lut[opponent]
            if d['total_number_of_samples'] > 20:
                # maybe set is_tight
                # is_tight = 1 if d['vpip'] < .28 else 0
                # is_aggressive = 1 if d['af'] > 1 else 0
                is_tight = d['vpip']
                is_aggressive = d['af']
                # maybe set is_aggressive
                hud[(offset * 2)] = is_tight
                hud[(offset * 2) + 1] = is_aggressive
            else:
                hud[(offset * 2) + 2] = 1  # is_balanced_or_unknown

        return obs.tolist() + [win_prob] + hud.tolist()

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
        # --- Step Environment with action --- #
        observations = []
        actions = []
        it = 0
        debug_action_list = []
        player_names = [p.name for p in players]
        player_names_not_folded = [pname for pname in player_names]
        fold_players = []
        target_players = []
        # in case we stop early because all relevant players have folded
        remaining_selected_players = []
        if self.fold_random_cards:
            # ActionGenOption.make_folds_from_top_players_with_randomized_hand: 2
            remaining_selected_players = [p.name for p in players if p.name in
                                          selected_players]
        else:
            if self.only_winners:
                # ActionGenOption.no_folds_top_player_only_wins: 1
                target_players = [p.name for p in players if p in episode.winners and
                                  p.name in selected_players]
                if not self.drop_folds:  # keep folds
                    # ActionGenOption.make_folds_from_showdown_loser_ignoring_rank: 3
                    target_players = [p.name for p in players if p in episode.winners]
                    fold_players = [p.name for p in players if
                                    p in episode.showdown_players and
                                    p not in episode.winners]
            else:
                # ActionGenOption.no_folds_top_player_all_showdowns: 0
                target_players = [p.name for p in players if p in episode.showdown_players
                                  and p.name in selected_players]
                if not self.drop_folds:  # keep folds
                    # target_players remain unchanged in this case
                    # ActionGenOption.make_folds_from_fish: 4
                    fold_players = [p.name for p in players if
                                    p in episode.showdown_players and
                                    p.name not in selected_players]
        if fold_players == target_players == remaining_selected_players == []:
            # case that selected player folded but we dont want folds in dataset
            assert not self.fold_random_cards, (
                "selected player folded but we dont want folds in dataset")
            return [], []
        while not done:
            try:
                action = action_list[it]
            except IndexError:
                raise self._EnvironmentDidNotTerminateInTimeError

            action_formatted = self.build_action(action)
            action_label = self.env.discretize(action_formatted)
            player_who_acted = action.who
            if action_label == ActionSpace.FOLD:
                player_names_not_folded.remove(player_who_acted)
            for player in players:
                if player.name == player_who_acted:
                    if player.name in remaining_selected_players:
                        # todo: recompute random cards postflop to bet good cards
                        #  preflop random cards are bad cards
                        if action_label == ActionSpace.FOLD:
                            # if obs[AugmentObservationFeatureColomns.Round_preflop]:
                            # resample player hole cards starting with buton range
                            # 40 % and co 35 mp and utg 25
                            # maybe? no! dont have time if win_prob > .5 resample again
                            pass
                        if self.use_hudstats:
                            obs = self.append_hud_stats(
                                obs,
                                player,
                                player_names,
                                n_opponents=len(player_names_not_folded) - 1)
                        observations.append(obs)
                        actions.append(action_label)
                        if action_label == ActionSpace.FOLD:
                            assert player.name not in target_players, (
                                "Player must not be in target_players")
                            remaining_selected_players.remove(player.name)
                    if player.name in target_players:
                        if self.use_hudstats:
                            obs = self.append_hud_stats(
                                obs,
                                player,
                                player_names,
                                n_opponents=len(player_names_not_folded) - 1)
                        observations.append(obs)
                        actions.append(action_label)

                    elif player.name in fold_players:
                        if self.use_hudstats:
                            obs = self.append_hud_stats(
                                obs,
                                player,
                                player_names,
                                n_opponents=len(player_names_not_folded) - 1)
                        observations.append(obs)
                        actions.append(ActionSpace.FOLD.value)

            debug_action_list.append(action_formatted)
            if not remaining_selected_players:
                if self.fold_random_cards:
                    return observations, actions
            obs, _, done, _ = self.env.step(action_formatted)
            it += 1

        if not observations:
            assert len(remaining_selected_players) == 1, (
                "big blind returned to player because every body folded")
            pname = remaining_selected_players[0]
            assert episode.players[pname].position == Positions6Max.BB, (
                "big blind returned to player because every body folded"
                "")
            # big blind returned to player because every body folded so they didnt get
            # to act
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

    def get_players_starting_with_first_mover(self, episode: PokerEpisodeV2) -> List[
        Player]:
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
        assert len(players_sorted) == len(
            episode.players), f'Failed for episode {episode.hand_id}'
        # [3,4,0,1,2] ->
        if num_players > 3:
            players_sorted = np.roll(players_sorted, -(num_players - 3))
        assert players_sorted[
                   0].seat_num_one_indexed == episode.btn_seat_num_one_indexed, f'Failed for episode {episode.hand_id}'
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
                # overwrite default hands with random cards that are not board or player cards
                idx0 = random.randint(0, len(deck) - 1)
                c0 = deck.pop(idx0)
                random_hand = [c0]
                idx1 = random.randint(0, len(deck) - 1)
                c1 = deck.pop(idx1)
                random_hand.append(c1)
                hands.append(random_hand)
        return hands

    @staticmethod
    def parse_action_gen_option(a_opt: ActionGenOption):
        only_winners = drop_folds = fold_random_cards = None
        if a_opt == ActionGenOption.no_folds_top_player_all_showdowns:
            only_winners = False
            drop_folds = True
        elif a_opt == ActionGenOption.no_folds_top_player_only_wins:
            only_winners = drop_folds = True
        elif a_opt == ActionGenOption.make_folds_from_top_players_with_randomized_hand:
            fold_random_cards = True
        elif a_opt == ActionGenOption.make_folds_from_showdown_loser_ignoring_rank:
            only_winners = True
            drop_folds = False
        elif a_opt == ActionGenOption.make_folds_from_fish:
            only_winners = drop_folds = False
        return only_winners, drop_folds, fold_random_cards

    def encode_episode(self,
                       episode: PokerEpisodeV2,
                       a_opt: ActionGenOption,
                       use_hudstats: bool,
                       selected_players: List[str],
                       limit_num_players: int = None,
                       verbose: bool = False) -> Tuple[
        Observations, Actions_Taken]:
        """Runs environment with steps from PokerEpisode.
                Returns observations and corresponding actions of players that made it to showdown."""
        try:
            self.use_hudstats = use_hudstats
            if self.use_hudstats:
                assert self.lut is not None
                assert len(selected_players) == 1
            only_winners, drop_folds, fold_random_cards = self.parse_action_gen_option(
                a_opt)
            self.verbose = verbose
            self.drop_folds = drop_folds
            self.fold_random_cards = fold_random_cards
            self._currency_symbol = episode.currency_symbol
            self.only_winners = only_winners
            skip_hand = True
            for pname in list(episode.players.keys()):
                if pname in selected_players:
                    skip_hand = False
            if skip_hand:
                return None, None
            try:
                players = self.get_players_starting_with_first_mover(episode)
            except AssertionError as e:
                logging.debug(e)
                logging.warning(e)
                return None, None
            if limit_num_players:
                if len(players) < limit_num_players:
                    return None, None
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
            if self.use_hudstats:
                self._feature_names = [col.name.lower() for col in
                                       list(FeaturesWithHudStats)]
            self.env.env.SMALL_BLIND = sb
            self.env.env.BIG_BLIND = bb
            self.env.env.ANTE = 0.0
            deck = np.full(shape=(13 * 4, 2), fill_value=Poker.CARD_NOT_DEALT_TOKEN_1D,
                           dtype=np.int8)
            board = make_board_cards(episode.board)
            if board:
                deck[:len(board)] = board
            else:
                assert not episode.actions[
                    'actions_flop'], "Board cards not allowed in games that ended preflop"
                assert not episode.actions[
                    'actions_turn'], "Board cards not allowed in games that ended preflop"
                assert not episode.actions[
                    'actions_river'], "Board cards not allowed in games that ended preflop"
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
            logging.debug(e)
            logging.warning(e)
            return None, None
        except Exception as e:
            logging.debug(e)
            logging.warning(e)
            return None, None
# in: .txt files
# out: .csv files? maybe npz or easier-on-memory formats preferred?
