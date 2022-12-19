"""PokerSnowie software internally stores played hands as text files.
Example:

#Game No : 1519972546
***** 888.de Snap Poker Hand History for Game 1519972546 *****
$0.01/$0.02 Blinds No Limit Holdem - *** 14 06 2022 00:01:43
Table Curico 6 Max (Real Money)
Seat 1 is the button
Total number of players : 6
Seat 1: UlyanaV ( $1.89 )
Seat 2: walterilmago ( $2.38 )
Seat 4: Lutzmolch ( $2 )
Seat 6: Dragusmen ( $3.23 )
Seat 7: rodrigohash ( $2.29 )
Seat 9: GMereuta ( $2.70 )
walterilmago posts small blind [$0.01]
Lutzmolch posts big blind [$0.02]
** Dealing down cards **
Dealt to Lutzmolch [ 4c, 3h ]
Dragusmen raises [$0.04]
rodrigohash folds
GMereuta raises [$0.16]
UlyanaV calls [$0.16]
walterilmago folds
Lutzmolch folds
Dragusmen calls [$0.12]
** Dealing flop ** [ 3s, Ks, 8c ]
Dragusmen checks
GMereuta bets [$0.25]
UlyanaV calls [$0.25]
Dragusmen folds
** Dealing turn ** [ 6c ]
GMereuta bets [$0.33]
UlyanaV calls [$0.33]
** Dealing river ** [ Kh ]
GMereuta checks
UlyanaV bets [$1.15]
GMereuta calls [$1.15]
** Summary **
UlyanaV shows [ 7h, 8h ]
GMereuta shows [ 6s, 4s ]
UlyanaV collected [ $3.72 ]

Our internal representation of played hands is given by PokerEpisode - instances.

This module is supposed to convert a `prl.baselines.supervised_learning.data_acquisition.core.parser.PokerEpisode` -
instance to .txt file for import in PokerSnowie.
"""
import datetime
from typing import List, Optional

from prl.baselines.pokersnowie.core.converter import PokerSnowieConverter
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode, Action, \
    ActionType

SnowieEpisode = str

action_types = ['folds', 'checks', 'calls', 'bets', 'raises']


class Converter888(PokerSnowieConverter):
    """Converts to PokerSnowie using 888-ExportFormat"""
    @staticmethod
    def parse_num(num):
        # parse string represenation of float, such that
        # it is rounded at most two digits
        # but only to non-zero decimal places
        # parse float
        if float(num).is_integer():
            return str(int(float(num)))
        else:
            ret = str(round(float(num), 2))
            decimals = str(round(float(num), 2)).split(".")[1]
            for i in range(2-len(decimals)):
                ret += "0"
            return ret

    def _convert_seats(self, episode):
        """
        Seat 1: UlyanaV ( $1.89 )
        Seat 2: walterilmago ( $2.38 )
        Seat 4: Lutzmolch ( $2 )
        Seat 6: Dragusmen ( $3.23 )
        Seat 7: rodrigohash ( $2.29 )
        Seat 9: GMereuta ( $2.70 )
        """
        ret = ""
        for seat in episode.player_stacks:
            stack = "$" + self.parse_num(seat.stack[1:])
            ret += f"{seat.seat_display_name}: {seat.player_name} ( {stack} )\n"
        return ret

    def _convert_blinds(self, episode):
        """
        walterilmago posts small blind [$0.01]
        Lutzmolch posts big blind [$0.02]
        """
        ret = ""
        for blind in episode.blinds:
            ret += blind.player_name + " posts " + blind.type + f" [{'$' + self.parse_num(blind.amount[1:])}]\n"
        return ret

    @staticmethod
    def _convert_dealt_cards(episode: PokerEpisode, hero_name: str):
        """
        ** Dealing down cards **
        Dealt to Lutzmolch [ 4c, 3h ]
        """
        ret = "** Dealing down cards **\n"
        for player in episode.showdown_hands:
            if player.name == hero_name:
                # '[As Th]' to '[ As, Th ]'
                cards = player.cards.replace(" ", ", ").replace("[", "[ ").replace("]", " ]")
                ret += f"Dealt to {hero_name} {cards}\n"
                return ret

    @staticmethod
    def _convert_community_cards(episode):
        """
        episode.board_cards == '[3s Ks 8c 6c Kh]'
        ** Dealing flop ** [ 3s, Ks, 8c ]
        ** Dealing turn ** [ 6c ]
        ** Dealing river ** [ Kh ]
        """
        flop_len = 10  # length of '[3s Ks 8c]'
        turn_len = 13  # length of '[3s Ks 8c 6c]'
        river_len = 16  # length of '[3s Ks 8c 6c Kh]'
        len_board = len(episode.board_cards)
        flop = "" if len_board < flop_len else f"[ {episode.board_cards[1:9].replace(' ', ', ')} ]"
        turn = "" if len_board < turn_len else f"[ {episode.board_cards[10:12]} ]"
        river = "" if len_board < river_len else f"[ {episode.board_cards[13:15]} ]"
        return {'flop': "" if len_board < flop_len else f"** Dealing flop ** {flop}\n",
                'turn': "" if len_board < turn_len else f"** Dealing turn ** {turn}\n",
                'river': "" if len_board < river_len else f"** Dealing river ** {river}\n"}

    @staticmethod
    def _convert_move_what(action: Action) -> str:
        """ActionType to ['folds', 'checks', 'calls', 'bets', 'raises']"""
        if action.action_type == ActionType.FOLD:
            return 'folds'
        elif action.action_type == ActionType.CHECK_CALL:
            if float(action.raise_amount) > 0:
                return 'calls'
            else:
                return 'checks'
        elif action.action_type == ActionType.RAISE:
            # todo if always returning 'bets' does not work the ntry 'raises'
            #  if that does not work either, then use previous actions to determine bet or raise
            return 'bets'

    def _convert_moves(self, episode):
        """
        Lutzmolch posts big blind [$0.02]
        ** Dealing down cards **
        Dealt to Lutzmolch [ 4c, 3h ]
        Dragusmen raises [$0.04]
        rodrigohash folds
        GMereuta raises [$0.16]
        UlyanaV calls [$0.16]
        walterilmago folds
        Lutzmolch folds
        Dragusmen calls [$0.12]
        ** Dealing flop ** [ 3s, Ks, 8c ]
        Dragusmen checks
        GMereuta bets [$0.25]
        UlyanaV calls [$0.25]
        Dragusmen folds
        ** Dealing turn ** [ 6c ]
        GMereuta bets [$0.33]
        UlyanaV calls [$0.33]
        ** Dealing river ** [ Kh ]
        GMereuta checks
        UlyanaV bets [$1.15]
        GMereuta calls [$1.15]
        ** Summary **
        """
        moves = {'preflop': '',
                 'flop': '',
                 'turn': '',
                 'river': ''}
        for a in episode.actions_total['as_sequence']:
            what = self._convert_move_what(a)
            how_much = f" [${self.parse_num(a.raise_amount)}]" if what not in ["folds", "checks"] else ""
            moves[a.stage] += f"{a.player_name} {what}{how_much}\n"
        return moves

    def _get_money_won(self, episode) -> dict:
        ret = {}
        for player in episode.money_collected:
            won = self.parse_num(player.collected[1:])
            ret[player.player_name] = won
        return ret

    def _convert_summary(self, episode):
        """
        ** Summary **
        Michaelcorb shows [ 9c, 6h ]
        Merodan shows [ Ad, 8d ]
        Merodan collected [ $0.04 ]
        """
        ret = "** Summary **\n"
        money_won = self._get_money_won(episode)
        for player in episode.showdown_hands:
            # '[As Th]' to '[ As, Th ]'
            hand = player.cards.replace(" ", ", ").replace("[", "[ ").replace("]", " ]")
            ret += f"{player.name} shows {hand}\n"
        for winner in episode.winners:
            # f" [{'$' + self.parse_num(blind.amount[1:])}]\n"
            ret += f"{winner.name} collected [ ${self.parse_num(money_won[winner.name])} ]\n"
        return ret

    def _from_poker_episode(self, episode: PokerEpisode, hero_name: str = None):  # -> SnowieEpisode:
        if len(episode.winners) > 2:
            # skip games where pot is split among more than two players
            return ""
        t = datetime.date.strftime(datetime.datetime.now(), '%d %m %y %H:%M:%S')
        sb = episode.blinds[0].amount[1:]
        bb = episode.blinds[1].amount[1:]
        btn = episode.player_stacks[episode.btn_idx]
        seats = self._convert_seats(episode)
        blinds = self._convert_blinds(episode)
        dealt_cards = self._convert_dealt_cards(episode, hero_name)
        community_cards: dict = self._convert_community_cards(episode)
        moves: dict = self._convert_moves(episode)
        summary = self._convert_summary(episode)
        episode_888 = f"#Game No : {episode.hand_id}\n" \
                      f"***** 888.de Snap Poker Hand History for Game {episode.hand_id} *****\n" \
                      f"${sb}/${bb} Blinds No Limit Holdem - *** {t}\n" \
                      f"Table Curico 6 Max (Real Money)\n" \
                      f"{btn.seat_display_name} is the button\n" \
                      f"Total number of players : {len(episode.player_stacks)}\n" + \
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
                      summary + \
                      "\n\n\n"
        return episode_888

    def from_poker_episode(self, episode: PokerEpisode, hero_names: Optional[List[str]] = None) -> List[SnowieEpisode]:
        """
        Converts episode to string representation that can be imported from PokerSnowie if written to txt file.
        Format used is 888 format.
        :param episode: Single hand played from start to finish
        :param hero_names: Player names which will be hero after conversion to snowie-episode. If none,
        one episode per player in showdown is created.
        :return: String representation of PokerEpisode that can be imported from PokerSnowie
        """
        if not hero_names:
            hero_names = [s.name for s in episode.showdown_hands]

        poker_snowie_episodes = []
        for hero in hero_names:
            poker_snowie_episodes.append(self._from_poker_episode(episode, hero))

        return poker_snowie_episodes
