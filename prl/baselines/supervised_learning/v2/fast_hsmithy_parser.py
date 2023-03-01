"""the old one uses python regexes which is unbearably slow for 250k files, let alone more
i tried making it better with multiprocessing but its still i/o bound so I have to improve on the
string processing side

I want a parser that can handle so much string data.
I want a parser that can handle incomplete episodes. I.e. if no showdown happened.
I need a data pipeline that is not unnecessarily complex
"""
import glob
import multiprocessing
import os
import re
import time
from functools import partial
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Generator

import numpy as np
import pandas as pd
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.base import ActionSpace
from prl.environment.Wrappers.utils import init_wrapped_env

from prl.baselines import DATA_DIR
from prl.baselines.evaluation.core.experiment import DEFAULT_DATE
from prl.baselines.supervised_learning.data_acquisition.core.encoder import Positions6Max
from prl.baselines.supervised_learning.data_acquisition.core.parser import \
    Action as ActionV1
from prl.baselines.supervised_learning.data_acquisition.core.parser import \
    PokerEpisode as PokerEpisodeV1, PlayerStack, \
    PlayerWithCards, PlayerWinningsCollected, Blind
from prl.baselines.supervised_learning.v2.config import top_100, top_20
from prl.baselines.supervised_learning.v2.new_txt_to_vector_encoder import EncoderV2
from prl.baselines.supervised_learning.v2.poker_model import Player, Action, \
    PokerEpisodeV2


# all the following functionality should be possible with only minimal parameterization (input_dir, output_dir, ...)
# 1. parse .txt files given list of players (only games containing players, or all if list is None)
# 2.


class ParseHsmithyTextToPokerEpisode:
    def __init__(self,
                 nl='NL50',
                 preflop_sep="*** HOLE CARDS ***",
                 flop_sep="*** FLOP ***",
                 turn_sep="*** TURN ***",
                 river_sep="*** RIVER ***",
                 showdown_sep="*** SHOW DOWN ***",
                 summary_sep="*** SUMMARY ***"):
        self.nl = nl
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
                                stack=round(stack * 100))
                players[pname] = player
            elif 'posts small blind' in line:
                sb_name: str = line.split(": ")[0]
                sb_amt = round(float(line.split(self.currency_symbol)[-1]) * 100)
                players[sb_name].position = Positions6Max.SB
                blinds['sb'] = sb_amt  # {sb_name: sb_amt}
            elif 'posts big blind' in line:
                bb_name: str = line.split(": ")[0]
                bb_amt = round(float(line.split(self.currency_symbol)[-1]) * 100)
                players[bb_name].position = Positions6Max.BB
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
                          how_much=amt)
        elif 'raises' in action:
            a = action.split('to ')[1].split(self.currency_symbol)[1]
            a = a.split(' and')[0]
            amt = round(float(a) * 100)
            return Action(who=pname, what=ActionSpace.RAISE_MIN_OR_THIRD_OF_POT,
                          how_much=amt)
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
        # if not '208958141851' in hand_str:
        #     return []
        # try:
        try:
            if '209160564676' in hand_str:
                print(hand_str)
            if '217918054212' in hand_str:
                print(hand_str)
            if '209160762232' in hand_str:
                print(hand_str)
            players, blinds = self.get_players_and_blinds(hand_str)
            info = self.rounds(hand_str)
            actions = self.get_actions(info)
            board_cards = ''
            showdown_players = []
            winners = []
            has_showdown = False
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
                        players[pname].money_won_this_round = amt
                    elif 'lost' in line:
                        # get money lost from actionsequence
                        money_contribution = 0
                        for action in actions['as_sequence']:
                            if action.who == pname:
                                assert action.what != ActionSpace.FOLD
                                money_contribution += action.how_much
                        players[pname].money_won_this_round = -money_contribution

        except Exception as e:
            return []
        for pname, player in players.items():
            if player.is_showdown_player:
                showdown_players.append(player)
                if player.money_won_this_round:
                    winners.append(player)
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
                    if not '*** SHOW DOWN ***' in hand and only_showdowns:
                        continue
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

    def parse_hand_histories(self) -> Generator[List[PokerEpisodeV2],None,None]:
        data_dir = os.path.join(DATA_DIR, *['01_raw', 'all_players', self.nl])
        assert os.path.exists(data_dir), "Must download data and unzip to " \
                                         "01_raw/all_players first"
        for f in glob.glob(data_dir + '**/*.txt'):
            try:
                episodes = self.parse_file(f)
                yield episodes
            except Exception:
                pass


def run_on_chunks(chunks):
    filenames = chunks
    parser = ParseHsmithyTextToPokerEpisode()
    env = init_wrapped_env(AugmentObservationWrapper,
                           [5000 for _ in range(6)],
                           blinds=(25, 50),
                           multiply_by=1, )
    encoder = EncoderV2(env)
    it = 0
    suffix = filenames[-1]
    filenames = filenames[:-1]

    while True:
        start = it * max_files_in_memory_at_once
        end = min((it + 1) * max_files_in_memory_at_once, n_files)
        if not filenames[start:end]:
            print(f'BREAK AT it={it}')
            break
        t0 = time.time()
        for player_name in top_100:
            training_data, labels = None, None
            for i, filename in enumerate(filenames[start:end]):
                print(f'Encoding file {i} / {n_files}')
                episodesV2 = parser.parse_file(filename)
                # convert episodes to PokerEpisodeV1
                # episodesV1 = [converter.convert_episode(ep) for ep in episodes]
                # episodes = None  # help gc
                # run rl_encoder

                for ep in episodesV2:
                    try:
                        observations, actions = encoder.encode_episode(ep,
                                                                       # drop_folds=False,
                                                                       drop_folds=True,
                                                                       only_winners=True,
                                                                       limit_num_players=5,
                                                                       randomize_fold_cards=True,
                                                                       selected_players=top_100,
                                                                       # selected_players=['ishuha'],
                                                                       verbose=True)
                    except Exception as e:
                        print(e)
                        continue
                    if not observations:
                        continue
                    if training_data is None:
                        training_data = observations
                        labels = actions
                    else:
                        try:
                            training_data = np.concatenate((training_data, observations),
                                                           axis=0)
                            labels = np.concatenate((labels, actions), axis=0)
                        except Exception as e:
                            print(e)
            print(
                f'Encoding {max_files_in_memory_at_once} files took {time.time() - t0} seconds.')
            if training_data is not None:
                columns = None
                header = False
                # file_path = os.path.abspath(f'./data_{it}.csv.bz2')
                # file_path = os.path.abspath(f'./top_100_only_wins_no_folds/data_{it}{suffix}.csv.bz2')
                file_path = os.path.abspath(
                    f'./top_100_only_wins_no_folds_per_player/{player_name}/data_{it}{suffix}.csv.bz2')
                if not os.path.exists(Path(file_path).parent):
                    os.makedirs(os.path.realpath(Path(file_path).parent), exist_ok=True)
                if not os.path.exists(file_path):
                    columns = encoder.feature_names
                    header = True
                df = pd.DataFrame(data=training_data,
                                  index=labels,
                                  # The index (row labels) of the DataFrame.
                                  columns=columns)
                # float to int if applicable
                df = df.apply(
                    lambda x: x.apply(lambda y: np.int8(y) if int(y) == y else y))
                # one hot encode button
                one_hot_btn = pd.get_dummies(df['btn_idx'], prefix='btn_idx')
                df = pd.concat([df, one_hot_btn], axis=1)
                df.drop('btn_idx', axis=1, inplace=True)
                df.to_csv(file_path,
                          index=True,
                          header=header,
                          index_label='label',
                          mode='a',
                          float_format='%.5f',
                          compression='bz2'
                          )
        it += 1
    return "Success."


def run_on_file(filename,
                out_dir,
                selected_players,
                drop_folds,
                only_winners,
                randomize_fold_cards,
                verbose=True,
                more_than_num_players=5,
                debug=False):
    parser = ParseHsmithyTextToPokerEpisode()
    env = init_wrapped_env(AugmentObservationWrapper,
                           [5000 for _ in range(6)],
                           blinds=(25, 50),
                           multiply_by=1, )
    encoder = EncoderV2(env)
    selected_players = [Path(filename).stem]
    if debug:
        selected_players = selected_players[:5]
    for player_name in selected_players:
        training_data, labels = None, None
        episodesV2 = parser.parse_file(filename)
        n_episodes = len(episodesV2)
        for i, ep in enumerate(episodesV2):
            print(f'Encoding episode no. {i}/{n_episodes}')
            try:
                observations, actions = encoder.encode_episode(ep,
                                                               # drop_folds=False,
                                                               drop_folds=drop_folds,
                                                               only_winners=only_winners,
                                                               limit_num_players=more_than_num_players,
                                                               randomize_fold_cards=randomize_fold_cards,
                                                               selected_players=selected_players,
                                                               # selected_players=['ishuha'],
                                                               verbose=verbose)
            except Exception as e:
                print(e)
                continue
            if not observations:
                continue
            if training_data is None:
                training_data = observations
                labels = actions
            else:
                try:
                    training_data = np.concatenate((training_data, observations), axis=0)
                    labels = np.concatenate((labels, actions), axis=0)
                except Exception as e:
                    print(e)
        if training_data is not None:
            columns = None
            header = False
            # file_path = os.path.abspath(f'./data_{it}.csv.bz2')
            # file_path = os.path.abspath(f'./top_100_only_wins_no_folds/data_{it}{suffix}.csv.bz2')
            file_path = os.path.abspath(
                f'{out_dir}/{player_name}/data.csv.bz2')
            if not os.path.exists(Path(file_path).parent):
                os.makedirs(os.path.realpath(Path(file_path).parent), exist_ok=True)
            if not os.path.exists(file_path):
                columns = encoder.feature_names
                header = True
            df = pd.DataFrame(data=training_data,
                              index=labels,  # The index (row labels) of the DataFrame.
                              columns=columns)
            # float to int if applicable
            df = df.apply(lambda x: x.apply(lambda y: np.int8(y) if int(y) == y else y))

            # one hot encode button
            one_hot_btn = pd.get_dummies(df['btn_idx'], prefix='btn_idx')
            df = pd.concat([df, one_hot_btn], axis=1)
            df.drop('btn_idx', axis=1, inplace=True)

            df.to_csv(file_path,
                      index=True,
                      header=header,
                      index_label='label',
                      mode='a',
                      float_format='%.5f',
                      compression='bz2'
                      )
    return "Success."


def make_dataset(unzipped_dir,
                 out_dir,
                 selected_players,
                 drop_folds,
                 only_winners,
                 randomize_fold_cards,
                 verbose=True,
                 more_than_num_players=5,
                 debug=False):
    filenames = glob.glob(unzipped_dir + "/**/*.txt", recursive=True)
    assert len(filenames) < 101
    start = time.time()
    # p = multiprocessing.Pool(20)
    if debug:
        p = multiprocessing.Pool(1)
    else:
        p = multiprocessing.Pool()
    # run f0
    run_fn = partial(run_on_file,
                     out_dir=out_dir,
                     selected_players=selected_players,
                     drop_folds=drop_folds,
                     only_winners=only_winners,
                     randomize_fold_cards=randomize_fold_cards,
                     verbose=verbose,
                     more_than_num_players=more_than_num_players,
                     debug=debug)
    for x in p.imap_unordered(run_fn, filenames):
        print(x + f'. Took {time.time() - start} seconds')
    print(f'Finished job after {time.time() - start} seconds.')

    p.close()


if __name__ == "__main__":
    """The new behaviour of the episode-encoder should be to
                     encode even non-showdown episodes. A set of selected players
                     is now mandatory. We choose the best 100 players.
                     We always use their actions as-they-are. This implies
                     we use all their games including non-showdown games.
                     When there is no showdown, we dont know their cards,
                     so we give them random cards and only use the observations
                     until they fold and end the game there."""
    unzipped_dir_to_S20 = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data_10"
    out_dir = "./results/top20/no_folds"
    debug = False
    # # make dataset DF2(20)
    # make_dataset(unzipped_dir=unzipped_dir_to_S20,
    #              out_dir=out_dir,
    #              selected_players=top_20,
    #              drop_folds=False,
    #              only_winners=False,
    #              randomize_fold_cards=True,
    #              verbose=True,
    #              debug=debug)

    # make dataset DNF1(20)
    make_dataset(unzipped_dir=unzipped_dir_to_S20,
                 out_dir=out_dir,
                 selected_players=top_20,
                 drop_folds=True,
                 only_winners=True,
                 randomize_fold_cards=True,
                 verbose=True,
                 debug=debug)

    # unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data_test"
    # unzipped_dir = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    # unzipped_dir = "/home/hellovertex/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/unzipped"
    # out_dir = "example.txt"
    # filenames = glob.glob(unzipped_dir + "/**/*.txt", recursive=True)
    # # parser = ParseHsmithyTextToPokerEpisode()
    # # converter = ConverterV2toV1()
    # # env = init_wrapped_env(AugmentObservationWrapper,
    # #                        [5000 for _ in range(6)],
    # #                        blinds=(25, 50),
    # #                        multiply_by=1, )
    # # encoder = EncoderV2(env)
    # max_files_in_memory_at_once = 1000
    # n_files = len(filenames)
    #
    # """ MULTIPROCESSING START """
    # x = 10000
    # chunks = []
    # current_chunk = []
    # i = 0
    # for file in filenames:
    #     current_chunk.append(file)
    #     if (i + 1) % x == 0:
    #         chunks.append(current_chunk)
    #         current_chunk = []
    #     i += 1
    # # trick to avoid multiprocessing writes to same file
    # for i, chunk in enumerate(chunks):
    #     chunk.append(f'{i}')
    # """ MP END """
    #
    # start = time.time()
    # # p = multiprocessing.Pool(20)
    # p = multiprocessing.Pool()
    # # run f0
    # for x in p.imap_unordered(run_on_chunks, chunks):
    #     print(x + f'. Took {time.time() - start} seconds')
    # print(f'Finished job after {time.time() - start} seconds.')
    #
    # p.close()
