"""the old one uses python regexes which is unbearably slow for 250k files, let alone more
i tried making it better with multiprocessing but its still i/o bound so I have to improve on the
string processing side

I want a parser that can handle so much string data.
I want a parser that can handle incomplete episodes. I.e. if no showdown happened.
I need a data pipeline that is not unnecessarily complex
"""
import glob
import os
import re
from typing import List, Tuple, Dict, Generator

from prl.environment.Wrappers.base import ActionSpace, ActionSpaceMinimal

from prl.baselines import DATA_DIR
from prl.baselines.supervised_learning.data_acquisition.core.encoder import Positions6Max
from prl.baselines.supervised_learning.v2.datasets.dataset_config import DatasetConfig
from prl.baselines.supervised_learning.v2.poker_model import Player, Action, \
    PokerEpisodeV2


class ParseHsmithyTextToPokerEpisode:
    def __init__(self,
                 dataset_config: DatasetConfig,
                 preflop_sep="*** HOLE CARDS ***",
                 flop_sep="*** FLOP ***",
                 turn_sep="*** TURN ***",
                 river_sep="*** RIVER ***",
                 showdown_sep="*** SHOW DOWN ***",
                 summary_sep="*** SUMMARY ***"):
        self.dataset_config = dataset_config
        self.preflop_sep = preflop_sep
        self.flop_sep = flop_sep
        self.turn_sep = turn_sep
        self.river_sep = river_sep
        self.summary_sep = summary_sep
        self.showdown_sep = showdown_sep

    def strip_next_round(self, strip_round, episode_str):
        return episode_str.split(strip_round)[0]

    def split_at_round(self, round, episode_str):
        try:
            return episode_str.split(round)[1]
        except IndexError:
            # index 1 cannot be accessed -> there is no `round`
            return ""

    def rounds(self, current_episode: str) -> Dict[str, str]:
        hole_cards = self.split_at_round(self.preflop_sep, current_episode)
        flop = self.split_at_round(self.flop_sep, current_episode)
        turn = self.split_at_round(self.turn_sep, current_episode)
        river = self.split_at_round(self.river_sep, current_episode)

        # strip each round from the other rounds, so we isolate them for stat updates
        # flop, turn, river may be empty string, so we have to find out where to
        # strip each round
        # 1. strip hole cards
        if flop:
            next_round = self.flop_sep
            hole_cards = self.strip_next_round(self.flop_sep, hole_cards)
        else:
            hole_cards = self.strip_next_round(self.summary_sep, hole_cards)
        # 2. strip flop cards
        if turn:
            # split at flop and strip from turn onwards
            flop = self.strip_next_round(self.turn_sep, flop)
        else:
            flop = self.strip_next_round(self.summary_sep, flop)
        # 3. strip turn cards
        if river:
            # split at turn and strip from river onwards
            turn = self.strip_next_round(self.river_sep, turn)
        else:
            turn = self.strip_next_round(self.summary_sep, turn)
        # 4. strip river cards
        if self.showdown_sep in river:
            river = self.strip_next_round(self.showdown_sep, river)
        else:
            river = self.strip_next_round(self.summary_sep, river)
        summary \
            = self.split_at_round(self.summary_sep, current_episode)

        # Assertions
        # PREFLOP
        assert self.flop_sep not in hole_cards
        assert self.turn_sep not in hole_cards
        assert self.river_sep not in hole_cards
        assert self.summary_sep not in hole_cards
        # FLOP
        assert self.preflop_sep not in flop
        assert self.turn_sep not in flop
        assert self.river_sep not in flop
        assert self.summary_sep not in flop
        # TURN
        assert self.preflop_sep not in turn
        assert self.flop_sep not in turn
        assert self.river_sep not in turn
        assert self.summary_sep not in turn
        # RIVER
        assert self.preflop_sep not in river
        assert self.flop_sep not in river
        assert self.turn_sep not in river
        assert self.summary_sep not in river

        return {'preflop': hole_cards,
                'flop': flop,
                'turn': turn,
                'river': river,
                'summary': summary}

    def get_players_and_blinds(self, hand_str) -> Tuple[
        Dict[str, Player], Dict[str, int]]:
        players = {}  # don't make this ordered, better to rely on names
        blinds = {}
        table = hand_str.split("*** HOLE CARDS ***")[0]
        lines = table.split('\n')
        self.currency_symbol = lines[0].split('No Limit (')[1][0]
        for line in lines:
            if "Table \'" in line:
                continue
            if 'Seat' in line:
                seat, player = line.split(': ')
                seat_num = int(seat[-1])
                pname = player.split(f'({self.currency_symbol}')[0].strip()
                stack = player.split(f'({self.currency_symbol}')[1].split(' in')[0]
                stack = float(stack)
                player = Player(name=pname,
                                seat_num_one_indexed=seat_num,
                                stack=round(stack * 100),
                                money_won_this_round=0)
                players[pname] = player
            elif 'posts small blind' in line:
                sb_name: str = line.split(": ")[0]
                sb_amt = round(float(line.split(self.currency_symbol)[-1]) * 100)
                players[sb_name].position = Positions6Max.SB
                players[sb_name].money_won_this_round = -sb_amt
                blinds['sb'] = sb_amt  # {sb_name: sb_amt}
            elif 'posts big blind' in line:
                bb_name: str = line.split(": ")[0]
                bb_amt = round(float(line.split(self.currency_symbol)[-1]) * 100)
                players[bb_name].position = Positions6Max.BB
                players[bb_name].money_won_this_round = -bb_amt
                blinds['bb'] = bb_amt  # {bb_name: bb_amt}
        num_players = len(players)
        return players, blinds

    def get_action(self, line):
        pname, action = line.split(': ')
        if 'folds' in action:
            return Action(who=pname, what=ActionSpace.FOLD, how_much=-1)
        elif 'checks' in action:
            return Action(who=pname, what=ActionSpace.CHECK_CALL, how_much=-1)
        elif 'calls' in action:
            a = action.split(self.currency_symbol)[1]
            a = a.split(' and')[0]
            amt = round(float(a) * 100)
            return Action(who=pname, what=ActionSpace.CHECK_CALL, how_much=amt)
        elif 'bets' in action:
            a = action.split(self.currency_symbol)[1]
            a = a.split(' and')[0]
            amt = round(float(a) * 100)
            return Action(who=pname, what=ActionSpace.RAISE_MIN_OR_THIRD_OF_POT,
                          how_much=amt,
                          info={'is_bet': True})
        elif 'raises' in action:
            a = action.split('to ')[1].split(self.currency_symbol)[1]
            a = a.split(' and')[0]
            amt = round(float(a) * 100)
            return Action(who=pname, what=ActionSpace.RAISE_MIN_OR_THIRD_OF_POT,
                          how_much=amt,
                          info={'is_bet': False})
        else:
            raise ValueError(f"Unknown action in {line}.")

    def _get_actions(self, lines, stage):
        lines = lines.split('\n')
        actions = []
        uncalled_bet = 0
        returned_to = None
        for line in lines:
            if 'Uncalled' in line:
                uncalled_bet = round(float(line.split(self.currency_symbol)[1][0]) * 100)
                returned_to = line.split('returned to ')[1]
                continue
            if not line:
                continue
            if not ':' in line:
                continue
            if 'said' in line:
                continue
            if "show hand" in line or 'shows' in line:
                continue
            if 'collected' in line:
                continue
            if 'leaves' in line:
                continue
            if 'joins' in line:
                continue
            action = self.get_action(line)
            action.stage = stage
            actions.append(action)
        return actions, uncalled_bet, returned_to

    def get_actions(self, info):
        actions_flop = []
        actions_turn = []
        actions_river = []
        actions_preflop, uncalled_bet, returned_to = self._get_actions(info['preflop'],
                                                                       'preflop')
        if not uncalled_bet:
            actions_flop, uncalled_bet, returned_to = self._get_actions(info['flop'],
                                                                        'flop')
        if not uncalled_bet:
            actions_turn, uncalled_bet, returned_to = self._get_actions(info['turn'],
                                                                        'turn')
        if not uncalled_bet:
            actions_river, uncalled_bet, returned_to = self._get_actions(info['river'],
                                                                         'river')
        as_sequence = []

        for actions in [actions_preflop, actions_flop, actions_turn, actions_river]:
            for action in actions:
                as_sequence.append(action)
        return {'actions_preflop': actions_preflop,
                'actions_flop': actions_flop,
                'actions_turn': actions_turn,
                'actions_river': actions_river,
                'as_sequence': as_sequence}, uncalled_bet, returned_to

    def blinds_folded(self, players: Dict[str, Player], actions: List[Action]):
        sb_first_action_was_fold = True
        bb_first_action_was_fold = True
        for action in actions:
            if players[action.who].position == Positions6Max.SB:
                if action.what == ActionSpaceMinimal.FOLD:
                    break
                else:
                    sb_first_action_was_fold = False
                    break
        for action in actions:
            if players[action.who].position == Positions6Max.BB:
                if action.what == ActionSpaceMinimal.FOLD:
                    break
                else:
                    bb_first_action_was_fold = False
                    break
        return sb_first_action_was_fold, bb_first_action_was_fold

    def update_money_won(self, players, blinds, actions, returned_to):
        # total_pot = blinds['sb'] + blinds['bb']  # included in total already
        total_pot = 0
        count_sb, count_bb = self.blinds_folded(players, actions['as_sequence'])
        if count_sb:
            total_pot += blinds['sb']
        if count_bb:
            total_pot += blinds['bb']
        for i, action in enumerate(actions['as_sequence']):
            amt = max(action.how_much, 0)
            total_pot += amt
            # supersimple2018: raises $37.50 to $50:
            # set money_won_this_round to +- $50 and not count previous bets/calls
            if action.what == ActionSpaceMinimal.RAISE:
                if action.info['is_bet']:
                    players[action.who].money_won_this_round -= amt
                else:
                    players[action.who].money_won_this_round = -amt
            else:
                players[action.who].money_won_this_round -= amt
        # players[returned_to].money_won_this_round += uncalled_bet
        if returned_to:
            players[returned_to].money_won_this_round += total_pot
        # else case is
    def make_showdown_cards(self, players: Dict[str, Player], info):
        board_cards = ''
        for line in info['summary'].split('\n'):
            if 'Board' in line:
                # Board [9d Th 3h 7d 6h]
                board_cards = line.split('Board ')[1]
            if 'showed' in line:
                has_showdown = True
                pname, cards = line.split(': ')[1].split('showed [')
                pname = pname.strip()
                if pname.endswith('(button)'):
                    pname = pname[:-8]
                pname = pname.strip()
                if pname.endswith('(small blind)'):
                    pname = pname[:-13]
                if pname.endswith('(big blind)'):
                    pname = pname[:-11]
                pname = pname.strip()
                cards = '[' + cards[:6]
                players[pname].cards = cards
                players[pname].is_showdown_player = True
                if ' and won ' in line:
                    try:
                        amt = line.split(f'({self.currency_symbol}')[1].split(')')[0]
                    except Exception as e:
                        print(line)
                        print(e)
                        raise e
                    amt = round(float(amt) * 100)
                    players[pname].money_won_this_round += amt
        return players, board_cards
    def parse_hand(self, hand_str):
        # if not '208958141851' in hand_str:
        #     return []
        # try:
        try:
            if '208959234900' in hand_str:
                a = 1
                print('debugme')
            players, blinds = self.get_players_and_blinds(hand_str)
            info = self.rounds(hand_str)
            actions, uncalled_bet, returned_to = self.get_actions(info)
            try:
                self.update_money_won(players, blinds, actions, returned_to)
            except Exception as e:
                print(e)
                return []
            showdown_players = []
            winners = []
            has_showdown = False
            players, board_cards = self.make_showdown_cards(players, info)
            for pname, player in players.items():
                if player.is_showdown_player:
                    has_showdown = True
                    showdown_players.append(player)
                    if player.money_won_this_round > 0:
                        winners.append(player)
        except Exception as e:
            print(e)
            return []
        btn_seat_num = int(hand_str.split('is the button')[0].strip()[-1])
        # except Exception as e:
        #     print(e)
        #     return []
        return PokerEpisodeV2(hand_id=int(hand_str.split(':')[0]),
                              currency_symbol=self.currency_symbol,
                              players=players,
                              blinds=blinds,
                              actions=actions,
                              board=board_cards,
                              has_showdown=has_showdown,
                              showdown_players=showdown_players,
                              winners=winners,
                              btn_seat_num_one_indexed=btn_seat_num)

    def parse_file(self, f: str, only_showdowns=False) -> List[PokerEpisodeV2]:
        """
        :param f: Absolute path to .txt file containing human-readable hhsmithy-export.
        :param out: Absolute path to .txt file containing
        :param filtered_players: If provided, the result will only contain games where
        a player in `filtered_players` participated.
        :param only_showdowns: If True, will only generate episodes that finished.
        As a consequence, if set to false, this returns PokerEpisodes where no player hands are visible.
        """
        # if filtered_players is None:
        # instead of using regex (which is slow) we must do it manually
        episodes = []
        try:
            with open(f, 'r',
                      encoding='utf-8') as f:  # pylint: disable=invalid-name,unspecified-encoding
                hand_database = f.read()
                hands_played = re.split(r'PokerStars Hand #', hand_database)[1:]

                for hand in hands_played:
                    # todo: remove `only_showdowns` -- always parse all games
                    #  filtering is applied during vectorization or preprocessing
                    # if not '*** SHOW DOWN ***' in hand and only_showdowns:
                    #     continue
                    if "leaves the table" in hand:
                        continue
                    if "sits out" in hand:
                        continue
                    parsed_hand = self.parse_hand(hand)
                    if parsed_hand:
                        episodes.append(parsed_hand)
        except Exception as e:
            print(e)
            return []
        return episodes

    @property
    def num_files(self):
        data_dir = os.path.join(DATA_DIR,
                                *['01_raw', self.dataset_config.nl, 'all_players'])
        return len(glob.glob(data_dir + '**/*.txt'))

    def parse_hand_histories_from_all_players(self) -> Generator[
        List[PokerEpisodeV2], None, None]:
        data_dir = os.path.join(DATA_DIR,
                                *['01_raw', self.dataset_config.nl, 'all_players'])
        assert os.path.exists(data_dir), "Must download data and unzip to " \
                                         "01_raw/all_players first"
        for f in glob.glob(data_dir + '**/*.txt'):
            try:
                episodes = self.parse_file(f)
                yield episodes
            except Exception:
                pass
