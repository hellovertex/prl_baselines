""" This module will
 - read .txt files inside ./data/
 - parse them to create corresponding PokerEpisode objects. """
import re
from typing import List, Tuple, Dict, Iterator, Iterable, Generator

from prl.baselines.supervised_learning.data_acquisition.core.parser import Parser, \
    PokerEpisode, Action, ActionType, \
    PlayerStack, Blind, PlayerWithCards, PlayerWinningsCollected

# REGEX templates
# PLAYER_NAME_TEMPLATE = r'([a-zA-Z0-9_.@#!-]+\s?[-@#!_.a-zA-Z0-9]*)'
# PLAYER_NAME_TEMPLATE = r'([óa-zA-Z0-9_.@#!-]+\s?[-@#!_.a-zA-Z0-9ó]*\s?[-@#!_.a-zA-Z0-9ó]*)'
# compile this with re.UNICODE to match any unicode char like é ó etc
PLAYER_NAME_TEMPLATE = r'([\w_.@#!-]+\s?[-@#!_.\w]*\s?[-@#!_.\w]*)'
STARTING_STACK_TEMPLATE = r'\(([$€￡Â£]+\d+.?\d*)\sin chips\)'
MATCH_ANY = r'.*?'  # not the most efficient way, but we prefer readabiliy (parsing is one time job)
POKER_CARD_TEMPLATE = r'[23456789TJQKAjqka][SCDHscdh]'
CURRENCY_SYMBOLS = ['$', '€', '￡',
                    'Â£']  # only these are currently supported, â‚¬ is € encoded


# ---------------------------- PokerStars-Parser ---------------------------------

class HSmithyParser(Parser):
    """Reads .txt files with poker games crawled from Pokerstars.com and parses them to
    PokerEpisodes."""

    class _InvalidPlayerNameError(ValueError):
        """We can encounter some weird player names like <'é=mc².Fin  é=mc³.Start'>
           We can parse unicode characters and very exotic names including those
           with multiple whitespaces but this name finally broke our nameparser
           Hence we skip these _very_ rare cases where the name is unparsable without
           additional efforts"""

    class _InvalidGameTypeError(ValueError):
        """We can encounter games where not only small blind,
        big blind, and ante are posted, but that contain lines like
        <'player posts small & big blinds'>. We skip these games
        because our env does not support them."""

    class _PlayerLeavesDuringPotContributionError(ValueError):
        """Edge case that player leaves before rundown"""

    class _CurrencyNotSupportedError(ValueError):
        """We only parse EUR, USD, GBP games"""

    class _Utf8NotSupportedError(ValueError):
        """A _very_ small fraction of txt files encodes the €-sign as <â‚¬>.
        Since it would be extra effort to adjust the parser accordingly, we skip these games."""

    class _NoSmallAndBigBlindGameTypeError(ValueError):
        """Only games with a single small blind a single big blind are accepted.
        Text files included games with multiple small blinds. Maybe a bug in their crawler.
        We skip these games"""

    class _ShowDownHappenedButNoSummaryDataExists(ValueError):
        """ The file Alrischa-0.25-0.50-USD-NoLimitHoldem-PokerStars-11-12-2020.txt
        had a wrong summary after showdown, no winners were printed and so the parser returned
        empty lsit showdown players:
        *** SUMMARY ***
        Total pot $44.35 | Rake $2
        Board [8c 6s Kc Kd 7s]
        Seat 1: edmundtrebus (big blind) folded on the Flop
        Seat 2: XOZAIH folded before Flop (didn't bet)
        """

    class PlayerCashedOutBeforeAwarded(ValueError):
        """When the message (pot not awarded as player cashed out) appears,
        the player cashed out before collecting the pot and we skip this game (happens very rarely)"""

    def __init__(self):
        # todo consider making HSmithyParser another abstract class and make derived PokerStars-Parser
        self._variant = None

    @staticmethod
    def get_hand_id(episode: str) -> int:
        """Returns the episode number of the current hand played. """
        pattern = re.compile(r'^(\d+):')
        return int(pattern.findall(episode)[0])

    @staticmethod
    def get_showdown(episode: str):
        """Return True if the current episode does not have a showdown.
        Args:
            episode: string representation of played episode as gotten from .txt files.
            Episode is assumed to contain showdown.
        Returns:
        """
        hands_played = re.split(r'\*\*\* SHOW DOWN \*\*\*', episode)
        assert len(hands_played) == 2, \
            f"Splitting showdown string went wrong: splits are {hands_played} "
        if 'mucked' in hands_played[1]:
            a = 1
        return hands_played[1]

    def get_winner(self, showdown: str) -> Tuple[
        List[PlayerWithCards], List[PlayerWithCards]]:
        """Return player name of player that won showdown.
        Specifically, the part after ... *** SUMMARY *** is parsed, which looks like:
        "*** SUMMARY ***
        Total pot $3 | Rake $0.15
        Board [Ac Js 9h 6d Tc]
        Seat 1: edmundtrebus folded before Flop (didn't bet)
        Seat 2: XOZAIH (button) folded before Flop (didn't bet)
        Seat 4: macfaelan (small blind) showed [4s Ad] and won ($2.85) with a pair of Aces
        Seat 6: Jay.Jay.1175 (big blind) mucked"
        """
        re_showdown_hands = re.compile(
            rf'Seat \d: {PLAYER_NAME_TEMPLATE}{MATCH_ANY} showed (\[{POKER_CARD_TEMPLATE} {POKER_CARD_TEMPLATE}])',
            re.UNICODE)
        re_winner = re.compile(
            rf'Seat \d: {PLAYER_NAME_TEMPLATE}{MATCH_ANY} showed (\[{POKER_CARD_TEMPLATE} {POKER_CARD_TEMPLATE}]) and won',
            re.UNICODE)
        showdown_hands = re_showdown_hands.findall(showdown)
        winners = re_winner.findall(showdown)
        # for hand in showdown_hands:
        #     if 'mucked' in hand[0] or 'mucked' in hand[1]:
        #         print('debug')
        #         a = 1
        # remove whitespaces in name field
        showdown_hands = [PlayerWithCards(name=hand[0].strip(), cards=hand[1])
                          for hand in showdown_hands]
        winners = [PlayerWithCards(name=hand[0].strip(), cards=hand[1])
                   for hand in winners]
        if not showdown_hands:
            raise self._ShowDownHappenedButNoSummaryDataExists()
        return winners, showdown_hands

    @staticmethod
    def _get_action_type(line: str):
        """Returns either 'fold', 'check_call', or 'raise with how much was called/raised or -1 in case of fold"""
        default_raise_amount = -1  # for fold, check and call actions
        if 'raises' in line or 'bets' in line:
            pattern = re.compile(r'(\d+\.?\d*)')
            raise_amount = pattern.findall(line)[-1]
            return ActionType.RAISE, raise_amount
        if 'calls' in line or 'checks' in line:
            if 'calls' in line:
                pattern = re.compile(r'(\d+\.?\d*)')
                amt = pattern.findall(line)[-1]
            else:
                amt = default_raise_amount
            return ActionType.CHECK_CALL, amt
        if 'folds' in line:
            return ActionType.FOLD, default_raise_amount
        raise RuntimeError(f"Could not parse action type from line: \n{line}")

    @staticmethod
    def get_actions(action_history: str, stage: str) -> List[Action]:
        """This is best explained by an example. Consider the string
          '''jimjames32: raises $4 to $6\n
          HHnguyen15: raises $14 to $20\n
          Pierson2323 joins the table at seat #5\n
          poppy20633 joins the table at seat #6\n
          3JackOFF: folds\n
          jimjames32: calls $14'''

          Each valid action follows the pattern {PLAYERNAME}: {action}\n
          So we split each line by ':', and check, which of the splitresults has exactly two elements
          (playername, action).
        """
        possible_actions = [possible_action.split(':') for possible_action in
                            action_history.split('\n')]
        actions = []
        for maybe_action in possible_actions:
            if len(maybe_action) == 2:
                action_type = maybe_action[1]
                # skip actions like 'Kaya1113 said, ":'(" which happens to have length 2 as well
                is_valid = [a_type for a_type in
                            ['folds', 'checks', 'calls', 'bets', 'raises'] if
                            (a_type in action_type)]
                if not is_valid:
                    continue
                action_type, raise_amount = HSmithyParser._get_action_type(action_type)
                action = Action(player_name=maybe_action[0],
                                action_type=action_type,
                                raise_amount=raise_amount,
                                stage=stage)
                actions.append(action)
        return actions

    @staticmethod
    def get_button(episode: str) -> int:
        """Returns the buttons seat number as displayed to user.
            Args:
                :episode string representation of played episode as gotten from .txt files
            Returns:
                button: int representing the seat number as displayed to user
        """
        ptn_button = re.compile(r"Seat #\d is the button")
        # ptn_button2 = re.compile(r"Seat #(\d) is the button")
        # button2 = int(ptn_button2.findall(s)[0])
        button = ptn_button.findall(episode)[0][6]
        return int(button)  # 1-indexed

    def get_player_stacks(self, line: str):
        """Returns stacks for each player.
            Args:
                :episode string representation of played episode as gotten from .txt files
            Returns:
                Example: [('Seat 1', 'jimjames32', '$82 '),
                          ('Seat 2', 'HHnguyen15', '$96.65'),
                          ('Seat 4', 'kjs609', '$200 ')]
        """
        # pattern = re.compile(rf"(Seat \d): {PLAYER_NAME_TEMPLATE}\s\(([$€]\d+.?\d*)\sin chips\)")
        pattern = re.compile(
            rf"(Seat \d): {PLAYER_NAME_TEMPLATE}\s{STARTING_STACK_TEMPLATE}", re.UNICODE)
        amounts = re.compile(rf'{STARTING_STACK_TEMPLATE}')
        stacks = pattern.findall(line)
        if not len(stacks) == len(amounts.findall(line)):
            raise self._InvalidPlayerNameError(
                "This error is raised, when we encountered a very exotic player name "
                "that cant be parsed by re.Unicode, like 'é=mc².Fin  é=mc³.Start' ")
        return pattern.findall(line)

    def get_blinds(self, episode: str) -> List[Tuple[str]]:
        """Returns blinds for current hand.
        Args:
            :episode string representation of played episode as gotten from .txt files
        Returns:
            Example: [('HHnguyen15', 'small blind', '$1'), ('kjs609', 'big blind', '$2')]
        """
        # pattern = re.compile(r"([a-zA-Z0-9]+): posts (small blind|big blind) ([$€]\d+.?\d*)")

        if "posts small & big blinds" in episode:
            raise self._InvalidGameTypeError

        pattern = re.compile(
            rf"{PLAYER_NAME_TEMPLATE}: posts (small blind|big blind) ([$€Â£￡]\d+.?\d*)",
            re.UNICODE)
        blinds = pattern.findall(episode)

        if not len(blinds) == 2:
            raise self._NoSmallAndBigBlindGameTypeError(
                f"Given blinds are invalid: {blinds}")

        return blinds

    @staticmethod
    def get_btn_idx(player_stacks: List[PlayerStack], btn_seat_num: int):
        """Returns seat index (not seat number) of seat that is currently the Button.
        Seats can be ["Seat 1", "Seat3", "Seat 5"]. If "Seat 5" is the Button,
        btn_idx=2 will be returned.
            Args:
                :player_stacks list of player info after parsing .txt files
            Returns:
                int index of button
        """
        # determine btn_idx
        for i, player_stack in enumerate(player_stacks):
            if int(player_stack.seat_display_name[5]) == btn_seat_num:
                return i
        raise RuntimeError(
            "Button index could not be determined. Guess we have to do more debugging...")

    @staticmethod
    def get_board_cards(episode: str):
        """Returns a string with board cards, e.g. '[6h Ts Td 9c Jc]'. """
        summary = episode.split("*** SUMMARY ***")
        pattern = re.compile(r'Board\s(\[.*?])\n')
        return pattern.findall(summary[1])[0]

    def _parse_actions(self, episode: str) -> Dict[str, List[Action]]:
        """Returns a dictionary with actions per stage.
          Args:
            episode: string representation of played episode as gotten from .txt files
          Returns:
            Dictionary with actions per stage:
            {'preflop': actions_preflop,
                'flop': actions_flop,
                'turn': actions_turn,
                'river': actions_river,
                'as_sequence': as_sequence}
        """
        hole_cards = episode.split("*** HOLE CARDS ***")[1].split("*** FLOP ***")[0]
        flop = episode.split("*** FLOP ***")[1].split("*** TURN ***")[0]
        turn = episode.split("*** TURN ***")[1].split("*** RIVER ***")[0]
        river = episode.split("*** RIVER ***")[1].split("*** SHOW DOWN ***")[0]

        actions_preflop = self.get_actions(hole_cards, stage='preflop')
        actions_flop = self.get_actions(flop, stage='flop')
        actions_turn = self.get_actions(turn, stage='turn')
        actions_river = self.get_actions(river, stage='river')
        as_sequence = actions_preflop + actions_flop + actions_turn + actions_river
        return {'preflop': actions_preflop,
                'flop': actions_flop,
                'turn': actions_turn,
                'river': actions_river,
                'as_sequence': as_sequence}

    def get_currency_symbol(self, episode: str):
        header = episode.split("\n")[0]
        for sbl in CURRENCY_SYMBOLS:
            if sbl in header:
                return sbl
        raise self._CurrencyNotSupportedError("Currency symbol not supported")

    @staticmethod
    def get_ante(currency_symbol: str, episode: str):
        pattern = re.compile(r'.*? posts the ante ([$€￡]\d+.?\d*)\n')
        res_ante = pattern.findall(episode)
        if res_ante:
            # every player posts the same ante
            assert res_ante[0] == res_ante[
                -1], "First and last player posted different amount of ante"
            return res_ante[0]
        return currency_symbol + '0.00'

    def get_money_collected(self, episode) -> List[PlayerWinningsCollected]:
        pattern = re.compile(rf"{PLAYER_NAME_TEMPLATE} collected ([$€Â£￡]\d+.?\d*)",
                             re.UNICODE)
        rake = re.compile(rf"Rake ([$€Â£￡]\d+.?\d*)").findall(episode)[0]
        collected = []
        for found in pattern.findall(episode):
            collected.append(
                PlayerWinningsCollected(player_name=found[0].split("\n")[-1],
                                        collected=found[1], rake=rake))
        if not collected or "cashed out" in episode:
            raise self.PlayerCashedOutBeforeAwarded()
        return collected

    def _parse_episode(self, episode: str, showdown: str) -> PokerEpisode:
        """UnderConstruction"""
        # edge case that player leaves before rundown should be skipped
        if "leaves the table" in episode:
            raise self._PlayerLeavesDuringPotContributionError

        hand_id = self.get_hand_id(episode)
        # if hand_id == 223221309065:
        #     print('need breakpoint here')

        currency_symbol = self.get_currency_symbol(episode)
        winners, showdown_hands = self.get_winner(showdown)
        # blinds = self.get_blinds(episode)
        blinds = [Blind(*blind) for blind in self.get_blinds(episode)]
        ante = self.get_ante(currency_symbol, episode)
        btn = self.get_button(episode)
        player_stacks = [PlayerStack(*stack) for stack in self.get_player_stacks(episode)]
        if not player_stacks:
            raise self._Utf8NotSupportedError(
                "utf8 encoded currency symbols are currently not supported")
        num_players = len(player_stacks)
        btn_idx = self.get_btn_idx(player_stacks, btn)
        board_cards = self.get_board_cards(episode)
        actions_total = self._parse_actions(episode)
        money_collected = self.get_money_collected(episode)
        return PokerEpisode(date='',  # todo
                            hand_id=hand_id,
                            variant=self._variant,
                            currency_symbol=currency_symbol,
                            num_players=num_players,
                            blinds=blinds,
                            ante=ante,
                            player_stacks=player_stacks,
                            btn_idx=btn_idx,
                            board_cards=board_cards,
                            actions_total=actions_total,
                            winners=winners,
                            showdown_hands=showdown_hands,
                            money_collected=money_collected,
                            info={'episode_str': episode})

    def parse_episode(self, episode: str) -> PokerEpisode:
        if not '*** SHOW DOWN ***' in episode:
            return []

        # get showdown
        showdown = self.get_showdown(episode)

        # todo: remove this -- requires changes in showdown hand parsing
        # skip if player did not show hand
        if 'mucks' in showdown:
            return []

        try:
            return self._parse_episode(episode, showdown)
        except self._InvalidPlayerNameError as e:
            # if an _InvalidPlayerNameError is thrown, we have encountered some weird player name like
            #  é=mc².Fin  é=mc³.Start
            # we can parse unicode characters and very exotic player names including those
            # with multiple whitespaces but this name finally broke our nameparser
            # Hence we skip these _very_ rare cases where the name is unparsable without further efforts
            a = 1
            return []
        except self._InvalidGameTypeError as e:
            # We can encounter games where not only small blind,
            # big blind, and ante are posted, but that contain lines like
            # <'player one posts small & big blinds'>. We skip these games
            # because our env does not support them.
            a = 1
            return []
        except self._PlayerLeavesDuringPotContributionError:
            # Edge case that player leaves before rundown
            a = 1
            return []
        except self._CurrencyNotSupportedError:
            # Only parse EUR, USD, GBP games
            a = 1
            return []
        except self._Utf8NotSupportedError:
            # A _very_ small fraction of txt files encodes the €-sign as <â‚¬>.
            # Since it would be extra effort to adjust the parser accordingly, we skip these games.
            a = 1
            return []
        except self._NoSmallAndBigBlindGameTypeError:
            # Only games with a single small blind a single big blind are accepted.
            # Text files included games with multiple small blinds. Maybe a bug in their crawler.
            # We skip these games.
            a = 1
            return []
        except self._ShowDownHappenedButNoSummaryDataExists:
            a = 1
            return []
        except self.PlayerCashedOutBeforeAwarded:
            a = 1
            return []

    def _parse_hands(self, hands_played) -> List[PokerEpisode]:
        parsed_hands = []
        for current in hands_played:  # c for current_hand
            # Only parse hands that went to Showdown stage, i.e. were shown
            a = 1
            if not '*** SHOW DOWN ***' in current:
                continue

            # get showdown
            showdown = self.get_showdown(current)

            # todo: remove this -- requires changes in showdown hand parsing
            # skip if player did not show hand
            if 'mucks' in showdown:
                continue

            try:
                hand = self._parse_episode(current, showdown)
                parsed_hands.append(hand)
            except self._InvalidPlayerNameError as e:
                # if an _InvalidPlayerNameError is thrown, we have encountered some weird player name like
                #  é=mc².Fin  é=mc³.Start
                # we can parse unicode characters and very exotic player names including those
                # with multiple whitespaces but this name finally broke our nameparser
                # Hence we skip these _very_ rare cases where the name is unparsable without further efforts
                a = 1
                continue
            except self._InvalidGameTypeError as e:
                # We can encounter games where not only small blind,
                # big blind, and ante are posted, but that contain lines like
                # <'player one posts small & big blinds'>. We skip these games
                # because our env does not support them.
                a = 1
                continue
            except self._PlayerLeavesDuringPotContributionError:
                # Edge case that player leaves before rundown
                a = 1
                continue
            except self._CurrencyNotSupportedError:
                # Only parse EUR, USD, GBP games
                a = 1
                continue
            except self._Utf8NotSupportedError:
                # A _very_ small fraction of txt files encodes the €-sign as <â‚¬>.
                # Since it would be extra effort to adjust the parser accordingly, we skip these games.
                a = 1
                continue
            except self._NoSmallAndBigBlindGameTypeError:
                # Only games with a single small blind a single big blind are accepted.
                # Text files included games with multiple small blinds. Maybe a bug in their crawler.
                # We skip these games.
                a = 1
                continue
            except self._ShowDownHappenedButNoSummaryDataExists:
                a = 1
                continue
            except self.PlayerCashedOutBeforeAwarded:
                a = 1
                continue
        return parsed_hands

    def parse_file(self, file_path) -> List[PokerEpisode]:
        self._variant = 'NoLimitHoldem'  # todo parse variant from filename
        with open(file_path, 'r',
                  encoding='utf-8') as f:  # pylint: disable=invalid-name,unspecified-encoding
            hand_database = f.read()
            hands_played = re.split(r'PokerStars Hand #', hand_database)[1:]
            return self._parse_hands(hands_played)
