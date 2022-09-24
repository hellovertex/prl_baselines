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

This module is supposed to convert a `PokerEpisode` - instance to .txt file for import in PokerSnowie.
"""
import datetime
from datetime import datetime as dt
from typing import NamedTuple

from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode


class SnowieEpisode(NamedTuple):
    date: str
    game_id: int

def _convert_seats():
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
     """

def from_poker_episode(episode: PokerEpisode, hero_name: str = None):  # -> SnowieEpisode:
    if not hero_name:
        hero_name = episode.winners[0].name
    for p in episode.player_stacks:
        p.
    str_seats = f"Seat 0 snowie1 100\n" \
                f"Seat 1 snowie2 100\n" \ 
                f"Seat 2 hero 100\n" \ 
                f"Seat 3 snowie3 100\n" \ 
                f"Seat 4 snowie4 100\n" \ 
                f"Seat 5 snowie5 100\n"

    snowie_episode = f"GameStart\n" \
                     f"PokerClient: ExportFormat\n" \
                     f"Date: {datetime.date.strftime(datetime.date.today(), '%d/%m/%y')}\n" \
                     f"TimeZone: GMT\n" \
                     f"Time: {dt.now().hour}:{dt.now().minute}:{dt.now().second}\n" \
                     f"GameId: {str(episode.hand_id)}\n" \
                     f"GameType: NoLimit\n" \
                     f"GameCurrency: {episode.currency_symbol}\n" \
                     f"SmallBlindStake: {episode.blinds[0].amount[1:]}" \
                     f"BigBlindStake: {episode.blinds[1].amount[1:]}" \
                     f"AnteStake: {episode.ante[1:]}" \
                     f"TableName: Table" \
                     f"Max number of player: 6" \
                     f"MyPlayerName: hero" \
                     f"DealerPosition: {episode.btn_idx}"
    return snowie_episode


def export(snowie_episode):
    pass
