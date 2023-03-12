import enum

from prl.environment.Wrappers.aoh import Positions6Max as pos

ranges = {
    # see www.bestpokercoaching.com/6max-preflop-chart
    'KK+': [(11, 11), (12, 12)],
    'QQ+': [(10, 10), (11, 11), (12, 12)],
    'JJ+': [(9, 9), (10, 10), (11, 11), (12, 12)],
    'QQ': [(10, 10)],
    'JJ': [(9, 9)],
    'TT': [(8, 8)],
    '99': [(7, 7)],
    'AK': [(12, 11), (11, 12)],
    'AQs': [(12, 10)],
    'AQo': [(10, 12)],
    'AJ': [(12, 9), (9, 12)],
    'AT': [(12, 8), (8, 12)],
    'AJs': [(12, 9)],
    'ATs': [(12, 8)],
    'AT-AQ': [(12, 8), (12, 9), (12, 10), (8, 12), (9, 12), (10, 12)],
    'A9s': [(12, 7)],
    'ATs+': [(12, 8), (12, 9), (12, 10), (12, 11)],
    'KQ': [(11, 10), (10, 11)],
    'KQs': [(11, 10)],
    'KJs': [(11, 9)],
    'KT+': [(11, 8), (11, 9), (11, 10), (8, 11), (9, 11), (10, 11)],
    'QT+': [(10, 8), (10, 9), (10, 11), (8, 10), (9, 10), (11, 10)],  # include KT+?
    'QJs': [(10, 9)],
    'JTs': [(9, 8)],
    '88-JJ': [(6, 6), (7, 7), (8, 8), (9, 9)],
    '66-TT': [(4, 4), (5, 5), (6, 6), (7, 7), (8, 8)],
    '55-QQ': [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)],
    '44-QQ': [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)],
    '22-JJ': [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),
              (9, 9)],
    '22-QQ': [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),
              (9, 9), (10, 10)],
    '56s': [(4, 3)],
    '67s': [(5, 4)],
    '68s': [(6, 4)],
    '78s': [(6, 5)],
    '79s': [(7, 5)],
    '89s': [(7, 6)],
    'T9s': [(8, 7)],
    'T7s': [(8, 5)],
    'T6s': [(8, 4)],
    '95s-96s': [(7, 3), (7, 4)],
    'A2s-A5s': [(12, 0), (12, 1), (12, 2), (12, 3)],
    'A2s-A8s': [(12, 0), (12, 1), (12, 2), (12, 3), (12, 4), (12, 5), (12, 6)],
    'A2s-A9s': [(12, 0), (12, 1), (12, 2), (12, 3), (12, 4), (12, 5), (12, 6), (12, 7)],
    'K5s-K8s': [(11, 3), (11, 4), (11, 5), (11, 6)],
    'Q5s-Q7s': [(10, 3), (10, 4), (10, 5)],
    'J5s-J7s': [(9, 3), (9, 4), (9, 5)],
    '44-AA': [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10),
              (11, 11), (12, 12)],
    '98s': [(7, 6)],
    'J9s': [(9, 7)],
    'JTs-ATs': [(9, 8), (10, 8), (11, 8), (12, 8)],
    'QJs-AJs': [(10, 9), (11, 9), (12, 9)],
    'KQs+': [(11, 10), (12, 10)],
    'AQ': [(12, 10), (10, 12)],
    'T8s-K8s': [(11, 6), (10, 6), (9, 6), (8, 6)],
    'Q9s-K9s': [(11, 7), (10, 7)],
    'JTo-KTo': [(8, 9), (8, 10), (8, 11)],
    'QJo-KJo': [(9, 10), (9, 11)],
    'A8o-A9o': [(12, 6), (12, 7)],
    '22-33': [(0, 0), (1, 1)],
    '76s': [(5, 4)],
    '86s': [(6, 5)],
    '97s': [(7, 5)],
    'ATo-AJo': [(8, 12), (9, 12)],
    '65s-75s': [(5, 3), (4, 3)],
    '32s-A2s': [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0),
                (10, 0), (11, 0)],
    '43s-K3s': [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1),
                (11, 1)],
    '54s-K4s': [(3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2), (10, 2), (11, 2)],
    '85s-K5s': [(6, 3), (7, 3), (8, 3), (9, 3), (10, 3), (11, 3)],
    '96s-K6s': [(7, 4), (8, 4), (9, 4), (10, 4), (11, 4)],
    'T7s-K7s': [(8, 5), (9, 5), (10, 5), (11, 5)],
    'T9o-K9o': [(7, 8), (7, 9), (7, 10), (7, 11)],
    '98o-K8o': [(6, 7), (6, 8), (6, 9), (6, 10), (6, 11)],
    '87o-A7o': [(5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12)],
    '76o-A6o': [(4, 5), (4, 6), (4, 7), (4, 8), (4, 10), (4, 11), (4, 12)],  # drop J6o
    '65o-75o': [(3, 4), (3, 5)],
    'K5o-A5o': [(3, 11), (3, 12)],
    'A2o-A4o': [(0, 12), (1, 12), (2, 12)],
    'KTs': [(11, 8)]
}


class HandRangePercentile(enum.IntEnum):
    PERCENTILE_13_50 = 0
    PERCENTILE_17_35 = 1
    PERCENTILE_26_00 = 2
    PERCENTILE_67_00 = 3


class HandRange:
    def sample_from_percentile(self, hand_range: HandRangePercentile):
        # vpip - pfr
        # prozent von 1326
        # out: c0, c1
        pass

    percentile_13_50 = ranges['44-AA'] + ranges['98s'] + ranges['J9s'] + ranges['T9s'] + \
                       ranges['JTs-ATs'] + ranges['QJs-AJs'] + ranges['KQ'] + ranges[
                           'AQ'] + ranges['AK'] + ranges['ATo-AJo']


# ranges['98s+'] = [ranges['JTs+'] + [()]]
open_raising_ranges = {pos.UTG: ranges['44-AA'] +
                                ranges['98s'] +
                                ranges['J9s'] +
                                ranges['T9s'] +
                                ranges['JTs-ATs'] +
                                ranges['QJs-AJs'] +
                                ranges['KQ'] +
                                ranges['AQ'] +
                                ranges['AK'] +
                                ranges['ATo-AJo']}

open_raising_ranges[pos.MP] = open_raising_ranges[pos.UTG] + \
                              ranges['22-33'] + \
                              ranges['76s'] + \
                              ranges['86s'] + \
                              ranges['A2s-A9s']
open_raising_ranges[pos.CO] = open_raising_ranges[pos.MP] + \
                              ranges['T8s-K8s'] + \
                              ranges['Q9s-K9s'] + \
                              ranges['JTo-KTo'] + \
                              ranges['QJo-KJo'] + \
                              ranges['A8o-A9o'] + \
                              ranges['65s-75s'] + \
                              ranges['86s'] + ranges['97s']
open_raising_ranges[pos.BTN] = open_raising_ranges[pos.CO] + \
                               ranges['32s-A2s'] + \
                               ranges['43s-K3s'] + \
                               ranges['54s-K4s'] + \
                               ranges['85s-K5s'] + \
                               ranges['96s-K6s'] + \
                               ranges['T7s-K7s'] + \
                               ranges['T9o-K9o'] + \
                               ranges['98o-K8o'] + \
                               ranges['87o-A7o'] + \
                               ranges['76o-A6o'] + \
                               ranges['65o-75o'] + \
                               ranges['K5o-A5o'] + ranges['A2o-A4o']
open_raising_ranges[pos.SB] = open_raising_ranges[pos.BTN]
open_raising_ranges[pos.BB] = open_raising_ranges[pos.CO]

vs_1_raiser_call = {}
vs_1_raiser_call[pos.MP] = {pos.UTG: ranges['55-QQ'] + ranges['AK'] + ranges['AQs']}
vs_1_raiser_call[pos.CO] = {pos.UTG: vs_1_raiser_call[pos.MP][pos.UTG] + ranges['KQs'],
                            pos.MP: vs_1_raiser_call[pos.MP][pos.UTG] + ranges['KQs']}
vs_1_raiser_call[pos.BTN] = {
    pos.UTG: ranges['44-QQ'] + ranges['AK'] + ranges['AQ'] + ranges['AJs'] + ranges[
        'KQs'],
    pos.MP: ranges['44-QQ'] + ranges['AK'] + ranges['AQ'] + ranges['AJs'] + ranges['KQs'],
    pos.CO: ranges['22-JJ'] + ranges['ATs+'] + ranges['AQo'] + ranges['KQs']}
vs_1_raiser_call[pos.SB] = {pos.UTG: ranges['22-QQ'] + ranges['AK'] + ranges['AQ'],
                            pos.MP: ranges['22-QQ'] + ranges['AK'] + ranges['AQ'],
                            pos.CO: ranges['22-JJ'] + ranges['AQ'] + ranges['AJ'] +
                                    ranges['ATs'] + ranges['KQs'] +
                                    ranges['KJs'] + ranges['QJs'] + ranges['JTs'],
                            pos.BTN: ranges['22-JJ'] + ranges['AQ'] + ranges['AJ'] +
                                     ranges['AT'] + ranges['A9s'] +
                                     ranges['KQ'] + ranges['KJs'] + ranges['KTs'] +
                                     ranges['QJs'] + ranges['JTs']}
vs_1_raiser_call[pos.BB] = {pos.UTG: vs_1_raiser_call[pos.SB][pos.CO],
                            pos.MP: vs_1_raiser_call[pos.SB][pos.CO],
                            pos.CO: vs_1_raiser_call[pos.SB][pos.CO],
                            pos.BTN: ranges['22-JJ'] + ranges['AT-AQ'] + ranges['A9s'] +
                                     ranges['KT+'] + ranges['QT+'] +
                                     ranges['JTs'],
                            pos.SB: open_raising_ranges[pos.CO]}
vs_1_raiser_3b_and_allin = {}
vs_1_raiser_3b_and_allin[pos.MP] = {pos.UTG: ranges['KK+']}
vs_1_raiser_3b_and_allin[pos.CO] = {pos.UTG: ranges['KK+'],
                                    pos.MP: ranges['KK+']}
vs_1_raiser_3b_and_allin[pos.BTN] = {pos.UTG: ranges['KK+'],
                                     pos.MP: ranges['KK+'],
                                     pos.CO: ranges['QQ+'] + ranges['AK']}
vs_1_raiser_3b_and_allin[pos.SB] = {pos.UTG: ranges['KK+'],
                                    pos.MP: ranges['KK+'],
                                    pos.CO: ranges['QQ+'] + ranges['AK'],
                                    pos.BTN: ranges['QQ+'] + ranges['AK']}
vs_1_raiser_3b_and_allin[pos.BB] = {pos.UTG: ranges['QQ+'],
                                    pos.MP: ranges['QQ+'],
                                    pos.CO: ranges['QQ+'],
                                    pos.BTN: ranges['QQ+'] + ranges['AK'],
                                    pos.SB: ranges['JJ+'] + ranges['AK'] + ranges['AQs']}
vs_1_raiser_3b_and_fold = {}
vs_1_raiser_3b_and_fold[pos.MP] = {pos.UTG: []}
vs_1_raiser_3b_and_fold[pos.CO] = {pos.UTG: [],
                                   pos.MP: []}
vs_1_raiser_3b_and_fold[pos.BTN] = {pos.UTG: ranges['56s'] +
                                             ranges['67s'] +
                                             ranges['78s'] +
                                             ranges['89s'] +
                                             ranges['T9s']}
vs_1_raiser_3b_and_fold[pos.BTN][pos.MP] = vs_1_raiser_3b_and_fold[pos.BTN][pos.UTG]
vs_1_raiser_3b_and_fold[pos.BTN][pos.CO] = vs_1_raiser_3b_and_fold[pos.BTN][pos.MP] + \
                                           ranges['A2s-A5s']
vs_1_raiser_3b_and_fold[pos.SB] = {pos.UTG: ranges['78s'] + ranges['89s'] + ranges['T9s'],
                                   pos.MP: ranges['78s'] + ranges['89s'] + ranges['T9s'],
                                   pos.CO: vs_1_raiser_3b_and_fold[pos.BTN][pos.CO],
                                   pos.BTN: vs_1_raiser_3b_and_fold[pos.BTN][pos.UTG]
                                            + ranges['A2s-A8s'] + ranges['68s'] + ranges[
                                                '79s']}
vs_1_raiser_3b_and_fold[pos.BB] = {pos.UTG: ranges['A2s-A9s'] +
                                            ranges['AK'] +
                                            vs_1_raiser_3b_and_fold[pos.BTN][pos.UTG]}
vs_1_raiser_3b_and_fold[pos.BB][pos.MP] = vs_1_raiser_3b_and_fold[pos.BB][pos.UTG]
vs_1_raiser_3b_and_fold[pos.BB][pos.CO] = vs_1_raiser_3b_and_fold[pos.BB][pos.MP]
vs_1_raiser_3b_and_fold[pos.BB][pos.BTN] = vs_1_raiser_3b_and_fold[pos.SB][pos.BTN]
vs_1_raiser_3b_and_fold[pos.BB][pos.SB] = ranges['K5s-K8s'] + \
                                          ranges['Q5s-Q7s'] + \
                                          ranges['J5s-J7s'] + \
                                          ranges['T6s'] + ranges['T7s'] + ranges[
                                              '95s-96s']

vs_3bet_after_openraise_4b_and_allin = {}
vs_3bet_after_openraise_4b_and_allin[pos.UTG] = {pos.MP: ranges['KK+'],
                                                 pos.CO: ranges['KK+'],
                                                 pos.BTN: ranges['QQ+'],
                                                 pos.SB: ranges['KK+'],
                                                 pos.BB: ranges['KK+']}
vs_3bet_after_openraise_4b_and_allin[pos.MP] = {pos.CO: ranges['KK+'],
                                                pos.BTN: ranges['QQ+'],
                                                pos.SB: ranges['KK+'],
                                                pos.BB: ranges['KK+']}

vs_3bet_after_openraise_4b_and_allin[pos.CO] = {pos.BTN: ranges['QQ+'] + ranges['AK'],
                                                pos.SB: ranges['QQ+'] + ranges['AK'],
                                                pos.BB: ranges['QQ+'] + ranges['AK']}
vs_3bet_after_openraise_4b_and_allin[pos.BTN] = {pos.SB: ranges['JJ+'] + ranges['AK'],
                                                 pos.BB: ranges['JJ+'] + ranges['AK']}
vs_3bet_after_openraise_4b_and_allin[pos.SB] = {pos.BB: ranges['JJ+'] + ranges['AK']}

vs_3bet_after_openraise_call = {}
vs_3bet_after_openraise_call[pos.UTG] = {pos.MP: ranges['QQ'] + ranges['AK'],
                                         pos.CO: ranges['JJ'] + ranges['QQ'] + ranges[
                                             'AK'],
                                         pos.BTN: ranges['TT'] + ranges['JJ'] + ranges[
                                             'AK'],
                                         pos.SB: ranges['TT'] + ranges['JJ'] + ranges[
                                             'QQ'] + ranges['AK'],
                                         pos.BB: ranges['TT'] + ranges['JJ'] + ranges[
                                             'QQ'] + ranges['AK']}
vs_3bet_after_openraise_call[pos.MP] = {
    pos.CO: ranges['TT'] + ranges['JJ'] + ranges['QQ'] + ranges['AK'],
    pos.BTN: ranges['99'] + ranges['TT'] + ranges['JJ'] + ranges['AK'],
    pos.SB: ranges['99'] + ranges['TT'] + ranges['JJ'] + ranges['QQ'] + ranges[
        'AK'],
    pos.BB: ranges['99'] + ranges['TT'] + ranges['JJ'] + ranges['QQ'] + ranges[
        'AK']}
vs_3bet_after_openraise_call[pos.CO] = {pos.BTN: ranges['88-JJ'],
                                        pos.SB: ranges['88-JJ'],
                                        pos.BB: ranges['88-JJ']}
vs_3bet_after_openraise_call[pos.BTN] = {
    pos.SB: ranges['66-TT'] + ranges['AQ'] + ranges['AJ'] + ranges['ATs'] + ranges['KQ'] +
            ranges['QJs'] + ranges[
                'JTs'],
    pos.BB: ranges['66-TT'] + ranges['AQ'] + ranges['AJ'] + ranges['ATs'] + ranges['KQ'] +
            ranges['QJs'] + ranges[
                'JTs']}
vs_3bet_after_openraise_call[pos.SB] = {
    pos.BB: ranges['66-TT'] + ranges['AQ'] + ranges['AJ'] + ranges['ATs'] + ranges['KQ'] +
            ranges['QJs'] + ranges[
                'JTs']}
