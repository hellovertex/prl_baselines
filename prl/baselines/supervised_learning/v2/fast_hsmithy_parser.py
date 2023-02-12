"""the old one uses python regexes which is unbearably slow for 250k files, let alone more
i tried making it better with multiprocessing but its still i/o bound so I have to improve on the
string processing side

I want a parser that can handle so much string data.
I want a parser that can handle incomplete episodes. I.e. if no showdown happened.
I need a data pipeline that is not unnecessarily complex
"""
import glob
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Union

from prl.environment.Wrappers.base import ActionSpace

from prl.baselines.evaluation.core.experiment import DEFAULT_DATE
from prl.baselines.supervised_learning.data_acquisition.core.encoder import Positions6Max
from prl.baselines.supervised_learning.data_acquisition.core.parser import PokerEpisode as PokerEpisodeV1, PlayerStack, \
    PlayerWithCards, PlayerWinningsCollected, Blind
from prl.baselines.supervised_learning.data_acquisition.core.parser import Action as ActionV1
from prl.baselines.supervised_learning.v2.new_txt_to_vector_encoder import Encoder

# all the following functionality should be possible with only minimal parameterization (input_dir, output_dir, ...)
# 1. parse .txt files given list of players (only games containing players, or all if list is None)
# 2.

class UnparsableFileException(ValueError):
    """This is raised when the parser can not finish parsing a .txt file."""


class EmptyPokerEpisode:
    """Placeholder for an empty PokerEpisode"""
    pass


@dataclass
class Player:
    name: str
    seat_num_one_indexed: int
    stack: int
    position: Optional[Positions6Max] = None
    cards: Optional[str] = None
    is_showdown_player: Optional[bool] = None
    money_won_this_round: Optional[int] = None


@dataclass
class Action:
    who: str
    what: Union[Tuple, ActionSpace]
    how_much: int
    stage: Optional[str] = None  # 'preflop' 'flop' 'turn' 'river'
    info: Optional[Dict] = None


@dataclass
class PokerEpisode:
    hand_id: int
    currency_symbol: str
    players: Dict[str, Player]
    blinds: Dict[str, Dict[str, int]]  # sb/bb -> player -> amount
    board: Optional[str]
    actions: Dict[str, List[Action]]
    has_showdown: Optional[bool]
    showdown_players: Optional[List[Player]]
    winners: Optional[List[Player]]
    btn_seat_num_one_indexed: Optional[int]


class ParseHsmithyTextToPokerEpisode:
    def __init__(self,
                 preflop_sep="*** HOLE CARDS ***",
                 flop_sep="*** FLOP ***",
                 turn_sep="*** TURN ***",
                 river_sep="*** RIVER ***",
                 showdown_sep="*** SHOW DOWN ***",
                 summary_sep="*** SUMMARY ***"):
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
        assert not self.flop_sep in hole_cards
        assert not self.turn_sep in hole_cards
        assert not self.river_sep in hole_cards
        assert not self.summary_sep in hole_cards
        # FLOP
        assert not self.preflop_sep in flop
        assert not self.turn_sep in flop
        assert not self.river_sep in flop
        assert not self.summary_sep in flop
        # TURN
        assert not self.preflop_sep in turn
        assert not self.flop_sep in turn
        assert not self.river_sep in turn
        assert not self.summary_sep in turn
        # RIVER
        assert not self.preflop_sep in river
        assert not self.flop_sep in river
        assert not self.turn_sep in river
        assert not self.summary_sep in river

        return {'preflop': hole_cards,
                'flop': flop,
                'turn': turn,
                'river': river,
                'summary': summary}

    def get_players_and_blinds(self, hand_str) -> Tuple[Dict[str, Player], Dict[str, Dict[str, int]]]:
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
                seat_num = seat[-1]
                pname = player.split(f'({self.currency_symbol}')[0].strip()
                stack = player.split(f'({self.currency_symbol}')[1].split(' in')[0]
                stack = float(stack)
                player = Player(name=pname,
                                seat_num_one_indexed=seat_num,
                                stack=round(stack * 100))
                players[pname] = player
            elif 'posts small blind' in line:
                sb_name: str = line.split(": ")[0]
                sb_amt = round(float(line.split(self.currency_symbol)[-1]) * 100)
                players[sb_name].position = Positions6Max.SB
                blinds['sb'] = {sb_name: sb_amt}
            elif 'posts big blind' in line:
                bb_name: str = line.split(": ")[0]
                bb_amt = round(float(line.split(self.currency_symbol)[-1]) * 100)
                players[bb_name].position = Positions6Max.BB
                blinds['bb'] = {bb_name: bb_amt}
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
            return Action(who=pname, what=ActionSpace.RAISE_MIN_OR_3BB, how_much=amt)
        elif 'raises' in action:
            a = action.split('to ')[1].split(self.currency_symbol)[1]
            a = a.split(' and')[0]
            amt = round(float(a) * 100)
            return Action(who=pname, what=ActionSpace.RAISE_MIN_OR_3BB, how_much=amt)
        else:
            raise ValueError(f"Unknown action in {line}.")

    def _get_actions(self, lines, stage):
        lines = lines.split('\n')
        actions = []
        for line in lines:
            if not line:
                continue
            if not ':' in line:
                continue
            if 'said' in line:
                continue
            if "show hand" in line or 'shows' in line:
                continue
            if 'Uncalled' in line:
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
        return actions

    def get_actions(self, info):
        actions_preflop = self._get_actions(info['preflop'], 'preflop')
        actions_flop = self._get_actions(info['flop'], 'flop')
        actions_turn = self._get_actions(info['turn'], 'turn')
        actions_river = self._get_actions(info['river'], 'river')
        as_sequence = []

        for actions in [actions_preflop, actions_flop, actions_turn, actions_river]:
            for action in actions:
                as_sequence.append(action)
        return {'actions_preflop': actions_preflop,
                'actions_flop': actions_flop,
                'actions_turn': actions_turn,
                'actions_river': actions_river,
                'as_sequence': as_sequence}

    def parse_hand(self, hand_str):
        try:
            players, blinds = self.get_players_and_blinds(hand_str)
            info = self.rounds(hand_str)
            actions = self.get_actions(info)
            board_cards = ''
            showdown_players = []
            winners = []
            has_showdown = False
            for line in info['summary']:
                if 'Board' in line:
                    # Board [9d Th 3h 7d 6h]
                    board_cards = line.split('Board ')[1]
                if 'showed' in line:
                    has_showdown = True
                    pname = line.split(':')[1].split('(')[0].strip()
                    cards = line.split('showed ')[1].split(' and')[0]
                    players[pname].cards = cards
                    players[pname].is_showdown_player = True
                    if 'won' in line:
                        amt = line.split(f'({self.currency_symbol}')[0].split(')')[0]
                        amt = round(float(amt) * 1000)
                        players[pname].money_won_this_round = amt

            for pname, player in players.items():
                if player.is_showdown_player:
                    showdown_players.append(player)
                    if player.money_won_this_round:
                        winners.append(player)
            btn_seat_num = int(hand_str.split('is the button')[0].strip()[-1])
        except Exception:
            return []
        return PokerEpisode(hand_id=int(hand_str.split(':')[0]),
                            currency_symbol=self.currency_symbol,
                            players=players,
                            blinds=blinds,
                            actions=actions,
                            board=board_cards,
                            has_showdown=has_showdown,
                            showdown_players=showdown_players,
                            winners=winners,
                            btn_seat_num_one_indexed=btn_seat_num)

    def parse_file(self, f: str,
                   out: str,
                   filtered_players: Optional[Tuple[str]],
                   only_showdowns: bool) -> List[PokerEpisode]:
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
        with open(f, 'r', encoding='utf-8') as f:  # pylint: disable=invalid-name,unspecified-encoding
            hand_database = f.read()
            hands_played = re.split(r'PokerStars Hand #', hand_database)[1:]
            for hand in hands_played:
                parsed_hand = self.parse_hand(hand)
                if parsed_hand:
                    episodes.append(parsed_hand)
        return episodes


class ConverterV2toV1:

    def get_pstacks(self, episode) -> List[PlayerStack]:
        player_stacks = []
        for pname, pinfo in episode.players.items():
            stack = PlayerStack(
                seat_display_name=f'Seat {pinfo.seat_num_one_indexed}',
                player_name=pname,
                stack=pinfo.stack
            )
            player_stacks.append(stack)
        return player_stacks

    def get_actions_total(self, episode) -> Dict[str, List[ActionV1]]:
        actions_total = {'preflop': [],
                         'flop': [],
                         'turn': [],
                         'river': [],
                         'as_sequence': []}
        for stage in ['preflop', 'flop', 'turn', 'river']:
            for act in episode.actions[f'actions_{stage}']:
                actv1 = ActionV1(stage=stage,
                                 player_name=act.who,
                                 action_type=act.what,
                                 raise_amount=act.how_much)
                actions_total[stage].append(actv1)
                actions_total['as_sequence'].append(actv1)
        return actions_total

    def get_winners(self, episode) -> List[PlayerWithCards]:
        winners = []
        for player in episode.winners:
            winners.append(PlayerWithCards(player.name,
                                           player.cards))
        return winners

    def get_showdown_hands(self, episode) -> List[PlayerWithCards]:
        showdown_hands = []
        for player in episode.showdown_players:
            showdown_hands.append(PlayerWithCards(player.name,
                                                  player.cards))
        return showdown_hands

    def get_money_collected(self, episode) -> List[PlayerWinningsCollected]:
        money_collected = []
        for player in episode.winners:
            money_collected.append(PlayerWinningsCollected(player.name,
                                                           player.money_won_this_round))
        return money_collected

    def get_blinds(self, episode) -> List[Blind]:
        sb = episode.blinds['sb']
        bb = episode.blinds['bb']
        blinds = []
        for pname, amount in sb.items():
            blinds.append(Blind(pname, 'small blind', amount))
        for pname, amount in bb.items():
            blinds.append(Blind(pname, 'big blind', amount))
        return blinds

    def convert_episode(self, episode: PokerEpisode) -> PokerEpisodeV1:
        player_stacks = self.get_pstacks(episode)
        blinds = self.get_blinds(episode)
        actions_total = self.get_actions_total(episode)
        winners = self.get_winners(episode)
        showdown_hands = self.get_showdown_hands(episode)
        money_collected = self.get_money_collected(episode)
        return PokerEpisodeV1(
            date=DEFAULT_DATE,
            hand_id=episode.hand_id,
            variant='NoLimitHoldEm',
            currency_symbol=episode.currency_symbol,
            num_players=len(episode.players),
            blinds=blinds,  # todo: '$1' vs int(1) in encoder
            ante='$0.00',
            player_stacks=player_stacks,  # # todo: '$1' vs int(1) in encoder
            btn_idx=episode.btn_seat_num_one_indexed,  # todo this shouldnt be needed
            board_cards=episode.board,
            actions_total=actions_total,
            winners=winners,  # todo: maybe be empty list [], check implications
            showdown_hands=showdown_hands,  # todo: maybe be empty list [], check implications
            money_collected=money_collected  # todo int vs str
        )


if __name__ == "__main__":
    unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data_test"
    unzipped_dir = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    out_dir = "example.txt"
    filenames = glob.glob(unzipped_dir + "/**/*.txt", recursive=True)
    parser = ParseHsmithyTextToPokerEpisode()
    converter = ConverterV2toV1()
    encoder = Encoder()
    max_files_in_memory_at_once = 1
    n_files = len(filenames)
    it = 0
    while True:
        start = it * max_files_in_memory_at_once
        end = min((it + 1) * max_files_in_memory_at_once, n_files)
        if not filenames[start:end]:
            print(f'BREAK AT it={it}')
            break
        for filename in filenames[start:end]:
            episodes = parser.parse_file(filename, out_dir, None, True)
            # convert episodes to PokerEpisodeV1
            episodesV1 = [converter.convert_episode(ep) for ep in episodes]
            episodes = None  # help gc
            # run rl_encoder
            for ep in episodesV1:
                """The new behaviour of the episode-encoder should be to
                 encode even non-showdown episodes. A set of selected players
                 is now mandatory. We choose the best 100 players.
                 We always use their actions as-they-are. This implies
                 we use all their games including non-showdown games.
                 When there is no showdown, we dont know their cards,
                 so we give them random cards and only use the observations
                 until they fold and end the game there."""
                encoder.encode_episode(ep,
                                       drop_folds=False,
                                       randomize_fold_cards=True,
                                       selected_players=True,
                                       verbose=True)
            a = 1
            # write to .npz
        it += 1