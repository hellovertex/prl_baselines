# preflop equities
# flop pot odds
# assume ranges (consider making them stochastic) for post flop MC analysis
import numpy as np
from prl.environment.Wrappers.aoh import Positions6Max as pos
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as cols
from prl.environment.Wrappers.base import ActionSpace

from prl.baselines.evaluation.utils import get_reset_config, pretty_print
from prl.baselines.examples.examples_tianshou_env import make_default_tianshou_env
from prl.environment.Wrappers.vectorizer import Vectorizer

hand0 = []

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
    '22-JJ': [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)],
    '22-QQ': [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)],
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
    'K5-K8s': [(11, 3), (11, 4), (11, 5), (11, 6)],
    'Q5s-Q7s': [(10, 3), (10, 4), (10, 5)],
    'J5s-J7s': [(9, 3), (9, 4), (9, 5)],
    '44-AA': [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12)],
    '98s': [(7, 6)],
    'J9s': [(9, 7)],
    'JTs+': [(9, 8), (10, 8), (11, 8), (12, 8)],
    'QJs+': [(10, 9), (11, 9), (12, 9)],
    'KQs+': [(11, 10), (12, 10)],
    'AQ': [(12, 10), (10, 12)],
    'T8s-K8s': [(11, 6), (10, 6), (9, 6), (8, 6)],
    'Q9s-K9s': [(11, 7), (10, 7)],
    'JTo-KTo': [(8, 9), (8, 10), (8, 11)],
    'QJo-KJo': [(9, 10), (9, 11)],
    'A8o-A9o': [(12, 6), (12, 7)],

}
# ranges['98s+'] = [ranges['JTs+'] + [()]]
open_raising_ranges = {pos.UTG: [
    ranges['44-AA'] +
    ranges['98s'] +
    ranges['J9s'] +
    ranges['T9s'] +
    ranges['JTs+'] +
    ranges['QJs+'] +
    ranges['KQs+'] +
    ranges['AQ']
]}

open_raising_ranges[pos.MP] = open_raising_ranges[pos.UTG] + [(12, 0), (12, 1), (12, 2), (12, 3), (12, 4), (12, 5),
                                                              (12, 6), (12, 7), (0, 0), (1, 1), (5, 4), (6, 5)]
open_raising_ranges[pos.CO] = open_raising_ranges[pos.MP] + \
                              ranges['T8s-K8s'] + \
                              ranges['Q9s-K9s'] + \
                              ranges['JTo-KTo'] + \
                              ranges['QJo-KJo'] + \
                              ranges['A8o-A9o'] + \
                              [(4, 3), (5, 3), (6, 4), (7, 5)]  # 65s, 75s, 86s, 97s
open_raising_ranges[pos.BTN] = open_raising_ranges[pos.CO] + \
                               [(1, 0),
                                (2, 0),
                                (3, 0),
                                (4, 0),
                                (5, 0),
                                (6, 0),
                                (7, 0),
                                (8, 0),
                                (9, 0),
                                (10, 0),
                                (11, 0)] + [(2, 1),
                                            (3, 1),
                                            (4, 1),
                                            (5, 1),
                                            (6, 1),
                                            (7, 1),
                                            (8, 1),
                                            (9, 1),
                                            (10, 1),
                                            (11, 1)] + \
                               [(3, 2),
                                (4, 2),
                                (5, 2),
                                (6, 2),
                                (7, 2),
                                (8, 2),
                                (9, 2),
                                (10, 2),
                                (11, 2)] + \
                               [(6, 3),
                                (7, 3),
                                (8, 3),
                                (9, 3),
                                (10, 3),
                                (11, 3)] + \
                               [(7, 4),
                                (8, 4),
                                (9, 4),
                                (10, 4),
                                (11, 4)] + \
                               [(8, 5),
                                (9, 5),
                                (10, 5),
                                (11, 5)] + \
                               [(7, 8),  # T9o-K9o
                                (7, 9),
                                (7, 10),
                                (7, 11)] + \
                               [(6, 7),  # 98o-K8o
                                (6, 8),
                                (6, 9),
                                (6, 10),
                                (6, 11)] + \
                               [(5, 6),  # 87o-A7o
                                (5, 7),
                                (5, 8),
                                (5, 9),
                                (5, 10),
                                (5, 11),
                                (5, 12)] + \
                               [(4, 5),  # 76o-A6o
                                (4, 6),
                                (4, 7),
                                (4, 8),  # drop J6o
                                (4, 10),
                                (4, 11),
                                (4, 12)] + [
                                   (3, 4), (3, 5), (3, 11), (3, 12),
                                   (0, 12), (1, 12), (2, 12)
                               ]


# pos.UTG: [],
#     pos.MP: [],
#     pos.CO: [],
#     pos.BTN: [],
#     pos.SB: [],
#     pos.BB: []
class RuleBasedAgent:
    def __init__(self, num_players, normalization):
        # assume that number of players does not change during the game
        # this assumption is valid, because we refill each player stack
        # after each round, such that the number of players never decreases
        self.num_players = num_players
        self.normalization = normalization
        positions = {2: (pos.BTN, pos.BB),
                     3: (pos.BTN, pos.SB, pos.BB),
                     4: (pos.BTN, pos.SB, pos.BB, pos.CO),
                     5: (pos.BTN, pos.SB, pos.BB, pos.MP, pos.CO),
                     6: (pos.BTN, pos.SB, pos.BB, pos.UTG, pos.MP, pos.CO)}
        self.positions = positions[num_players]

    def get_raises_preflop(self, obs):
        r00 = obs[cols.Preflop_player_0_action_0_what_2]
        r01 = obs[cols.Preflop_player_0_action_1_what_2]
        r10 = obs[cols.Preflop_player_1_action_0_what_2]
        r11 = obs[cols.Preflop_player_1_action_1_what_2]
        r20 = obs[cols.Preflop_player_2_action_0_what_2]
        r21 = obs[cols.Preflop_player_2_action_1_what_2]
        r30 = obs[cols.Preflop_player_3_action_0_what_2]
        r31 = obs[cols.Preflop_player_3_action_1_what_2]
        r40 = obs[cols.Preflop_player_4_action_0_what_2]
        r41 = obs[cols.Preflop_player_4_action_1_what_2]
        r50 = obs[cols.Preflop_player_5_action_0_what_2]
        r51 = obs[cols.Preflop_player_5_action_1_what_2]
        return {
            0: [r00, r01],
            1: [r10, r11],
            2: [r20, r21],
            3: [r30, r31],
            4: [r40, r41],
            5: [r50, r51],
            'total': r00 + r01 + r10 + r11 + r20 + r21 + r30 + r31 + r40 + r41 + r50 + r51
        }

    def get_preflop_openraise_or_fold(self,
                                      obs,
                                      hand,
                                      hero_position):
        # bet/fold
        bb = obs[cols.Big_blind] * self.normalization
        if hero_position == pos.UTG:
            if hand in open_raising_ranges[pos.UTG]:
                # raise 4bb
                return 2, 4 * bb
            return ActionSpace.FOLD
        elif hero_position == pos.MP:
            if hand in open_raising_ranges[pos.MP]:
                # raise 3bb
                return 2, 3 * bb
            return ActionSpace.FOLD
        elif hero_position == pos.CO:
            if hand in open_raising_ranges[pos.CO]:
                # raise 2.5 bb
                return 2, 2.5 * bb
            return ActionSpace.FOLD
        elif hero_position == pos.BTN:
            if hand in open_raising_ranges[pos.BTN]:
                # raise 2bb
                return 2, 2 * bb
            return ActionSpace.FOLD
        elif hero_position == pos.SB:
            if hand in open_raising_ranges[pos.SB]:
                # raise 2bb
                return 2, 2 * bb
            return ActionSpace.FOLD
        elif hero_position == pos.BB:
            if hand in open_raising_ranges[pos.BB]:
                # raise 4bb
                return 2, 4 * bb
            return ActionSpace.FOLD
        else:
            raise ValueError(f"Invalid position of current player: {hero_position}")

    def get_preflop_3bet_or_call_or_fold(self, obs, hand, hero_position, raises):
        # if raiser was UTG
        if hero_position == pos.MP:
            pass
        elif hero_position == pos.CO:
            pass
        elif hero_position == pos.BTN:
            pass
        elif hero_position == pos.SB:
            pass
        elif hero_position == pos.BB:
            pass
        else:
            raise ValueError(f"Hero Position must be in [MP, CO, BTN, SB, BB] but was {hero_position}")

    def get_preflop_4bet_or_call_or_fold(self, obs, hand, hero_position, raises):
        pass

    def act(self, obs: np.ndarray, legal_moves):
        print('YEAY')
        c0 = obs[cols.First_player_card_0_rank_0:cols.First_player_card_0_suit_3 + 1]
        c1 = obs[cols.First_player_card_1_rank_0:cols.First_player_card_1_suit_3 + 1]
        r0, s0 = np.where(c0 == 1)[0]
        r1, s1 = np.where(c1 == 1)[0]
        max_r = max(r0, r1)
        min_r = min(r0, r1)
        hand = (max_r, min_r) if s0 == s1 else (min_r, max_r)
        btn_idx = np.where(obs[cols.Btn_idx_is_0:cols.Btn_idx_is_5 + 1] == 1)[0]

        a = 1
        hero_position = self.positions[-btn_idx % self.num_players]

        if obs[cols.Round_preflop]:
            # last two raises one-hot encoded for each player
            raises = self.get_raises_preflop(obs)
            # case no previous raise
            if raises['total'] == 0:
                return self.get_preflop_openraise_or_fold(obs,
                                                          hand,
                                                          hero_position)
            # case one previous raise:
            if raises['total'] == 1:
                # call/ xor 3b/ALLIN  xor 3b/FOLD (semi-bluff)
                return self.get_preflop_3bet_or_call_or_fold(obs,
                                                             hand,
                                                             hero_position,
                                                             raises)
            # case 3bet:
            if raises['total'] > 1:
                # 4b/ALLIN or Call but we dont semi-bluff
                pass
        elif obs[cols.Round_flop]:
            pass
        elif obs[cols.Round_turn]:
            pass
        elif obs[cols.Round_river]:
            pass
        else:
            raise ValueError("Observation does not contain round-bit. "
                             "It either is a terminal observation or not initialized")
        return 0


if __name__ == '__main__':
    num_players = 3
    verbose = True
    hidden_dims = [256]
    starting_stack = 5000
    stack_sizes = [starting_stack for _ in range(num_players)]
    agent_names = [f'p{i}' for i in range(num_players)]
    # rainbow_config = get_rainbow_config(default_rainbow_params)
    # RainbowPolicy(**rainbow_config).load_state_dict...
    # env = get_default_env(num_players, starting_stack)
    env = make_default_tianshou_env(mc_model_ckpt_path=None,  # dont use mc
                                    agents=agent_names,
                                    num_players=len(agent_names))
    normalization = env.env.env.env_wrapped.normalization
    agents = [RuleBasedAgent(num_players, normalization) for _ in range(num_players)]
    #  todo init from state dict and feed NN observations
    board = '[Ks Kh Kd Kc 2s]'
    player_hands = ['[Jh Jc]', '[4h 6s]', '[As 5s]']
    state_dict = get_reset_config(player_hands, board)
    assert len(agents) == num_players == len(stack_sizes)
    options = {'reset_config': state_dict}
    i = 0
    for epoch in range(4):
        obs = env.reset(options=options)
        agent_id = obs['agent_id']
        legal_moves = obs['mask']
        obs = obs['obs']
        while True:
            i = agent_names.index(agent_id)
            action = agents[i].act(obs, legal_moves)
            print(f'AGNET_ID = {agent_id}')
            pretty_print(i, obs, action)
            obs_dict, cum_reward, terminated, truncated, info = env.step(action)
            rews = cum_reward
            agent_id = obs_dict['agent_id']
            print(f'AGENT_ID', agent_id)
            obs = obs_dict['obs']
            print(f'GOT REWARD {cum_reward}')
            if terminated:
                break
