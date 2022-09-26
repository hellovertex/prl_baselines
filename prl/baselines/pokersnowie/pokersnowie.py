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

from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode, PlayerStack, Blind


class PokerSnowieEpisode:
    @staticmethod
    def _convert_seats(player_stacks: List[PlayerStack], hero_name: str) -> Tuple[str, Dict[str, str]]:
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
            name = hero_name if hero_name == p.player_name else f"snowie {ith_snowie_player}"
            stack = p.stack[1:]
            ret += f"Seat {seat_id} {name} {stack}\n"
            player_names_dict[p.player_name] = name
            ith_snowie_player += 1
        return ret, player_names_dict

    @staticmethod
    def _convert_blinds(blinds: List[Blind], player_names_dict: Dict[str, str]) -> str:
        ret = ""
        # player_names_dict[blinds[0].player_name]
        # todo

        return ret

    @staticmethod
    def _convert_dealt_cards(showdown_hands, player_names_dict):
        ret = ""
        # todo
        return ret

    @staticmethod
    def _convert_community_cards(board_cards):
        ret = {}
        # todo
        return {}

    @staticmethod
    def _convert_moves(actions_total, player_names_dict):
        ret = {}
        # todo
        return ret

    @staticmethod
    def _get_maybe_uncalled_bet(episode, player_names_dict):
        ret = ""
        # todo
        return ret

    def from_poker_episode(self, episode: PokerEpisode, hero_name: str = None):  # -> SnowieEpisode:
        if not hero_name:
            hero_name = episode.winners[0].name
        seats, player_names_dict = self._convert_seats(episode.player_stacks, hero_name)
        blinds = self._convert_blinds(episode.blinds, player_names_dict)
        dealt_cards = self._convert_dealt_cards(episode.showdown_hands, player_names_dict)
        community_cards: dict = self._convert_community_cards(episode.board_cards)
        moves: dict = self._convert_moves(episode.actions_total, player_names_dict)
        maybe_move_uncalled_bet = self._get_maybe_uncalled_bet(episode, player_names_dict)
        snowie_episode = f"GameStart\n" \
                         f"PokerClient: ExportFormat\n" \
                         f"Date: {datetime.date.strftime(datetime.date.today(), '%d/%m/%y')}\n" \
                         f"TimeZone: GMT\n" \
                         f"Time: {dt.now().hour}:{dt.now().minute}:{dt.now().second}\n" \
                         f"GameId: {str(episode.hand_id)}\n" \
                         f"GameType: NoLimit\n" \
                         f"GameCurrency: {episode.currency_symbol}\n" \
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
                         "GameEnd"
        return snowie_episode

    def export(self, snowie_episode):
        pass
