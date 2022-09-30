"""PokerSnowie software internally stores played hands as text files.
Example:

GameStart
PokerClient: ExportFormat
Date: 24/09/2022
TimeZone: GMT
Time: 14:20:39
GameId:40978514
GameType:NoLimit
GameCurrency: $
SmallBlindStake: 1
BigBlindStake: 2
AnteStake: 0
TableName: Table
Max number of players: 6
MyPlayerName: hero
DealerPosition: 5
Seat 0 snowie1 100
Seat 1 snowie2 100
Seat 2 hero 100
Seat 3 snowie3 100
Seat 4 snowie4 100
Seat 5 snowie5 100
SmallBlind: snowie1 1
BigBlind: snowie2 2
Dealt Cards: [Tc5c]
Move: hero folds 0
Move: snowie3 raise_bet 4
Move: snowie4 folds 0
Move: snowie5 folds 0
Move: snowie1 folds 0
Move: snowie2 call_check 2
FLOP Community Cards:[Jc 7c 2c]
Move: snowie2 raise_bet 2
Move: snowie3 call_check 2
TURN Community Cards:[Jc 7c 2c 5d]
Move: snowie2 call_check 0
Move: snowie3 call_check 0
RIVER Community Cards:[Jc 7c 2c 5d 3d]
Move: snowie2 raise_bet 3
Move: snowie3 raise_bet 41
Move: snowie2 folds 0
Move: snowie3 uncalled_bet 38
Winner: snowie3 19.00
GameEnd

Our internal representation of played hands is given by PokerEpisode - instances.

This module is supposed to convert a `prl.baselines.supervised_learning.data_acquisition.core.parser.PokerEpisode` -
instance to .txt file for import in PokerSnowie.
"""
import datetime
from datetime import datetime as dt
from typing import NamedTuple, List, Dict, Tuple

from prl.baselines.pokersnowie.converter import Converter
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode, PlayerStack, Blind

SnowieEpisode = str


class SnowieConverter(Converter):

    def _convert_seats(self, player_stacks: List[PlayerStack], hero_name: str) -> Tuple[str, Dict[str, str]]:
        """Internal representation:
        [PlayerStack(seat_display_name='Seat 1', player_name='Solovyova', stack='$50'),
         PlayerStack(seat_display_name='Seat 2', player_name='x elrubio x', stack='$64.22'),
         PlayerStack(seat_display_name='Seat 3', player_name='Salkin77', stack='$143.45'),
         PlayerStack(seat_display_name='Seat 4', player_name='JuanAlmighty', stack='$57.70'),
         PlayerStack(seat_display_name='Seat 5', player_name='igROCK90', stack='$50'),
         PlayerStack(seat_display_name='Seat 6', player_name='seibras81', stack='$29.61')]

         PokerSnowie representation:

        Seat 0 snowie1 100
        Seat 1 snowie2 100
        Seat 2 hero 100
        Seat 3 snowie3 100
        Seat 4 snowie4 100
        Seat 5 snowie5 100

        :returns
        ret: Pokersnowie string representation of seats-section
        player_names_dict: mapping from player names to poker snowie names
         """
        ret = ""
        ith_snowie_player = 1
        player_names_dict = {}
        for p in player_stacks:
            seat_id = int(p.seat_display_name[-1]) - 1
            name = "hero" if hero_name == p.player_name else f"snowie{ith_snowie_player}"
            stack = p.stack[1:]
            ret += f"Seat {seat_id} {name} {self.parse_num(stack)}\n"
            player_names_dict[p.player_name] = name
            if name != "hero":
                ith_snowie_player += 1
        return ret, player_names_dict

    def _convert_blinds(self, blinds: List[Blind], player_names_dict: Dict[str, str]) -> str:
        sb = blinds[0]
        bb = blinds[1]
        sb_name = player_names_dict[sb.player_name.split('\n')[-1]]
        bb_name = player_names_dict[bb.player_name.split('\n')[-1]]
        return f"SmallBlind: {sb_name} {self.parse_num(sb.amount[1:])}\nBigBlind: {bb_name} {self.parse_num(sb.amount[1:])}\n"

    @staticmethod
    def _convert_dealt_cards(showdown_hands, player_names_dict):
        cards = "Dealt Cards: "
        for k, v in player_names_dict.items():
            if v == 'hero':
                for player in showdown_hands:
                    if player.name == k:
                        return cards + player.cards.replace(" ", "") + "\n"

    @staticmethod
    def _convert_community_cards(board_cards):
        return {'flop': 'FLOP Community Cards:[' + board_cards[1:9] + ']\n',
                'turn': 'TURN Community Cards:[' + board_cards[1:12] + ']\n',
                'river': 'RIVER Community Cards:[' + board_cards[1:15] + ']\n'}

    def _convert_moves(self, actions_total, player_names_dict):
        moves = {'preflop': '',
                 'flop': '',
                 'turn': '',
                 'river': ''}
        for a in actions_total['as_sequence']:
            p_name = player_names_dict[a.player_name]
            move = ['folds', 'call_check', 'raise_bet'][a.action_type]
            amt = self.parse_num(str(a.raise_amount)) if float(a.raise_amount) > 0 else '0'
            moves[a.stage] += f'Move: {p_name} {move} {amt}\n'
        return moves

    def _convert_winners(self, episode: PokerEpisode, player_names_dict):
        player_money_in_pot = {}
        for name in player_names_dict.values():
            player_money_in_pot[name] = 0

        total_pot = 0

        # add blinds
        for blind in episode.blinds:
            p_name = player_names_dict[blind.player_name.split('\n')[-1]]
            amount = round(float(blind.amount[1:]), 2)
            player_money_in_pot[p_name] += amount
            total_pot += amount

        for a in episode.actions_total['as_sequence']:
            p_name = player_names_dict[a.player_name]
            if float(a.raise_amount) > 0:
                player_money_in_pot[p_name] += float(a.raise_amount)
                total_pot += float(a.raise_amount)
        biggest_contributor = max(player_money_in_pot, key=player_money_in_pot.get)
        biggest_contribution = player_money_in_pot.pop(biggest_contributor)
        second_biggest_or = max(player_money_in_pot, key=player_money_in_pot.get)
        second_biggest_tion = player_money_in_pot[second_biggest_or]
        result = ""
        if biggest_contribution > second_biggest_tion:
            diff = round(biggest_contribution - second_biggest_tion, 2)
            result += f"Move: {biggest_contributor} uncalled_bet {self.parse_num(str(diff))}\nWinner: {biggest_contributor} {self.parse_num(str(total_pot))}\n"
        else:  # showdown
            for showdown_hand in episode.showdown_hands:
                p_name = player_names_dict[showdown_hand.name]
                cards = showdown_hand.cards
                result += f"Showdown: {p_name} {cards}\n"
            for winner in episode.winners:
                result += f"Winner: {player_names_dict[winner.name]} {self.parse_num(str(total_pot))}\n"
        return result

    @staticmethod
    def parse_num(num: str):
        # parse string represenation of float, such that
        # it is rounded at most two digits
        # but only to non-zero decimal places
        # parse float
        num = round(float(num), 2)
        num = str(num).rstrip("0")
        if num.endswith("."):
            num = num[:].rstrip(".")
        return num

    def _from_poker_episode(self, episode: PokerEpisode, hero_name: str = None):  # -> SnowieEpisode:
        """
        Seat 0 snowie1 100
        Seat 1 snowie2 100
        Seat 2 hero 100
        Seat 3 snowie3 100
        Seat 4 snowie4 100
        Seat 5 snowie5 100
        """
        seats, player_names_dict = self._convert_seats(episode.player_stacks, hero_name)
        """
        SmallBlind: snowie1 1
        BigBlind: snowie2 2
        """
        blinds = self._convert_blinds(episode.blinds, player_names_dict)
        """
        Dealt Cards: [Tc5c]
        """
        dealt_cards = self._convert_dealt_cards(episode.showdown_hands, player_names_dict)
        """
        FLOP Community Cards:[Jc 7c 2c]
        TURN Community Cards:[Jc 7c 2c 5d]
        RIVER Community Cards:[Jc 7c 2c 5d 3d]
        """
        community_cards: dict = self._convert_community_cards(episode.board_cards)
        """
        Move: snowie3 folds 0
        Move: snowie4 raise_bet 4
        Move: snowie5 folds 0
        Move: snowie1 raise_bet 10
        Move: snowie2 folds 0
        Move: hero call_check 8
        Move: snowie4 folds 0
        """
        moves: dict = self._convert_moves(episode.actions_total, player_names_dict)
        """
        Move: hero uncalled_bet 16
        Winner: hero 16.00
        """
        if str(episode.hand_id) in ["233352635215", "233353992643", "223219756750", "233352891451"]:
            # todo rundowns not supported yet
            # todo: handle this scenario where showdown + uncalled_bet due to all in
            """Move: snowie3 call_check 53.5
            Move: hero uncalled_bet 5
            Showdown: hero [Js 8s]
            Showdown: snowie3 [Jc As]
            Winner: snowie3 199.00
            GameEnd"""
            return ""
        if str(episode.hand_id) in ['222455657718', '233352891451']:
            print('inspect winners')
        maybe_move_uncalled_bet = self._convert_winners(episode, player_names_dict)
        snowie_episode = f"GameStart\n" \
                         f"PokerClient: ExportFormat\n" \
                         f"Date: {datetime.date.strftime(datetime.date.today(), '%d/%m/%y')}\n" \
                         f"TimeZone: GMT\n" \
                         f"Time: {dt.now().hour}:{dt.now().minute}:{dt.now().second}\n" \
                         f"GameId:{str(episode.hand_id)}\n" \
                         f"GameType:NoLimit\n" \
                         f"GameCurrency: $\n" \
                         f"SmallBlindStake: {episode.blinds[0].amount[1:]}\n" \
                         f"BigBlindStake: {episode.blinds[1].amount[1:]}\n" \
                         f"AnteStake: {episode.ante[1:]}\n" \
                         f"TableName: Table\n" \
                         f"Max number of player: 6\n" \
                         f"MyPlayerName: hero\n" \
                         f"DealerPosition: {episode.btn_idx}\n" + \
                         seats + \
                         blinds + \
                         dealt_cards + \
                         moves['preflop'] + \
                         community_cards['flop'] + \
                         moves['flop'] + \
                         community_cards['turn'] + \
                         moves['turn'] + \
                         community_cards['river'] + \
                         moves['river'] + \
                         maybe_move_uncalled_bet + \
                         "GameEnd\n\n"
        return snowie_episode

    def from_poker_episode(self, episode: PokerEpisode, hero_names: List[str] = None) -> List[SnowieEpisode]:
        """
        Converts episode to string representation that can be imported from PokerSnowie if written to txt file.
        Uses PokerSnowies ExportFormat for episode-conversion.
        :param episode: Single hand played from start to finish
        :param hero_names: Player names which will be hero after conversion to snowie-episode.
        :return: String representation of PokerEpisode that can be imported from PokerSnowie if written to txt file
        """
        if not hero_names:
            hero_names = [s.name for s in episode.showdown_hands]

        poker_snowie_episodes = []
        for hero in hero_names:
            poker_snowie_episodes.append(self._from_poker_episode(episode, hero))

        return poker_snowie_episodes

    def export(self, snowie_episode):
        pass
