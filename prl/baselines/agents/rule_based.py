# preflop equities
# flop pot odds
# assume ranges (consider making them stochastic) for post flop MC analysis
import random

import numpy as np
from prl.environment.Wrappers.aoh import Positions6Max as pos
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as cols
from prl.environment.Wrappers.base import ActionSpace

from prl.baselines.evaluation.utils import get_reset_config, pretty_print
from prl.baselines.examples.examples_tianshou_env import make_default_tianshou_env
from prl.environment.Wrappers.vectorizer import Vectorizer


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
        ordered_positions = {2: (pos.BTN, pos.BB),
                             3: (pos.BTN, pos.SB, pos.BB),
                             4: (pos.CO, pos.BTN, pos.SB, pos.BB),
                             5: (pos.MP, pos.CO, pos.BTN, pos.SB, pos.BB),
                             6: (pos.UTG, pos.MP, pos.CO, pos.BTN, pos.SB, pos.BB)}
        self.positions = positions[num_players]
        self.ordered_positions = ordered_positions[num_players]
        self.open_calling_range_MP = ranges['55-QQ'] + ranges['AK'] + ranges['AQs']
        self.open_calling_range_CO = ranges['55-QQ'] + ranges['AK'] + ranges['AQ'] + ranges['KQs']
        self.open_calling_range_BTN = ranges['44-QQ'] + + ranges['AK'] + ranges['AQ'] + ranges['KQs'] + ranges['AJs']
        self.open_calling_range_BTN_VS_CO = ranges['22-JJ'] + ranges['ATs+'] + ranges['AQo'] + ranges['KQs']
        self.open_calling_range_SB_BB = ranges['22-JJ'] + ranges['AQ'] + ranges['AJ'] + ranges['ATs'] + ranges['KQs'] + \
                                        ranges['KJs'] + ranges['QJs'] + ranges['JTs']
        self.open_raising_sizes = (4, 3, 2.5, 2, 2, 4)
        # do not change this probability,
        # we need this for polarized 3b/Fold strategy vs 1 preflop Raiser
        # the bot folds after being c-bet on a semi-bluff, so it is fine to have .95% as the range for semi-bluffs
        # is very conservative
        self.semi_bluff_probability = .95
        self.common_semi_bluff_range_preflop = ranges['56s'] + ranges['67s'] + ranges['78s'] + ranges['89s'] + ranges[
            'T9s']
        self.raise_action = 2

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
        who_raised = []
        if r00 + r01 > 0:
            who_raised.append(0)
        if r10 + r11 > 0:
            who_raised.append(1)
        if r20 + r21 > 0:
            who_raised.append(2)
        if r30 + r31 > 0:
            who_raised.append(3)
        if r40 + r41 > 0:
            who_raised.append(4)
        if r50 + r51 > 0:
            who_raised.append(5)

        return {
            0: [r00, r01],
            1: [r10, r11],
            2: [r20, r21],
            3: [r30, r31],
            4: [r40, r41],
            5: [r50, r51],
            'total': r00 + r01 + r10 + r11 + r20 + r21 + r30 + r31 + r40 + r41 + r50 + r51,
            'who_raised': who_raised
        }

    def get_preflop_openraise_or_fold(self,
                                      obs,
                                      hand,
                                      hero_position):
        if hand in open_raising_ranges[hero_position]:
            bb = obs[cols.Big_blind] * self.normalization
            raise_amount = self.open_raising_sizes[hero_position.value] * bb
            return self.raise_action, raise_amount
        return ActionSpace.FOLD

    def vs_1_preflop_raiser(self, hand, hero_position, raises, btn_idx):
        # given raises (relative to observer) determine who open-raised
        # the following line is kept for reference but the assertion is wrong
        # in fact there can be multiple raises before hero gets first chance to raise
        # assert len(raises['who_raised']) == 1

        # raise_idx = raises['who_raised'][0]
        raise_idx = min(raises['who_raised'])
        assert raise_idx > 0, "Hero must not have raised already here."
        open_raiser_position = self.positions[(raise_idx - btn_idx) % self.num_players]

        # Open Raisor was UTG
        if hero_position == pos.MP:
            assert open_raiser_position == pos.UTG
            if hand in self.open_calling_range_MP:
                return 1, -1
            elif hand in ranges['KK+']:
                return ActionSpace.RAISE_POT

        # Open Raisor was UTG or Middle Position
        elif hero_position == pos.CO:
            assert open_raiser_position == pos.UTG or open_raiser_position == pos.MP
            if hand in self.open_calling_range_CO:
                return 1, -1
            elif hand in ranges['KK+']:
                return ActionSpace.RAISE_POT

        # Open Raisor was before BUTTON
        elif hero_position == pos.BTN:
            assert open_raiser_position in [pos.UTG, pos.MP, pos.CO]
            # BTN VS CO
            if open_raiser_position == pos.CO:
                if hand in self.open_calling_range_BTN_VS_CO:
                    return ActionSpace.CHECK_CALL
                elif hand in ranges['QQ+'] or hand in ranges['AK']:
                    return ActionSpace.RAISE_POT
                elif hand in self.common_semi_bluff_range_preflop or hand in ranges['A2s-A5s']:
                    if random.random() < self.semi_bluff_probability:
                        return ActionSpace.RAISE_POT
                return ActionSpace.FOLD
            # BTN VS UTG OR MP
            else:
                if hand in self.open_calling_range_BTN:
                    return ActionSpace.CHECK_CALL
                elif hand in ranges['KK+']:
                    return ActionSpace.RAISE_POT
                elif hand in self.common_semi_bluff_range_preflop:
                    if random.random() < self.semi_bluff_probability:
                        return ActionSpace.RAISE_POT
                return ActionSpace.FOLD

        # Open Raisor was before SB
        elif hero_position == pos.SB:
            assert open_raiser_position in [pos.UTG, pos.MP, pos.CO, pos.BTN]
            if open_raiser_position in [pos.UTG, pos.MP]:
                if hand in ranges['22-QQ'] or hand in ranges['AK'] or hand in ranges['AQ']:
                    return ActionSpace.CHECK_CALL
                elif hand in ranges['KK+']:
                    return ActionSpace.RAISE_POT
                elif hand in ranges['78s'] or hand in ranges['89s'] or hand in ranges['T9s']:
                    if random.random() < self.semi_bluff_probability:
                        return ActionSpace.RAISE_POT
                return ActionSpace.FOLD

            # CUTOFF OPENRAISED
            elif open_raiser_position == pos.CO:
                if hand in self.open_calling_range_SB_BB:
                    return ActionSpace.CHECK_CALL
                elif hand in ranges['QQ+'] or hand in ranges['AK']:
                    return ActionSpace.RAISE_POT
                elif hand in self.common_semi_bluff_range_preflop or hand in ranges['A2s-A5s']:
                    if random.random() < self.semi_bluff_probability:
                        return ActionSpace.RAISE_POT
                return ActionSpace.FOLD
            elif open_raiser_position == pos.BTN:
                if hand in self.open_calling_range_SB_BB or hand in ranges['AT'] or hand in ranges['A9s'] or hand in \
                        ranges['KQ'] or hand in ranges['KTs']:
                    return ActionSpace.CHECK_CALL
                elif hand in ranges['QQ+'] or hand in ranges['AK']:
                    return ActionSpace.RAISE_POT
                elif hand in self.common_semi_bluff_range_preflop or hand in ranges['A2s-A8s'] or hand in ranges[
                    '68s'] or hand in ranges['79s']:
                    if random.random() < self.semi_bluff_probability:
                        return ActionSpace.RAISE_POT
                return ActionSpace.FOLD

        # Open Raisor was before BB
        elif hero_position == pos.BB:
            assert open_raiser_position in [pos.UTG, pos.MP, pos.CO, pos.BTN, pos.SB]

            # VS UTG, MP, CO
            if open_raiser_position in [pos.UTG, pos.MP, pos.CO]:
                if hand in self.open_calling_range_SB_BB:
                    return ActionSpace.CHECK_CALL
                elif hand in ranges['QQ+']:
                    return ActionSpace.RAISE_POT
                elif hand in ranges['A2s-A9s'] or hand in ranges['AK'] or hand in self.common_semi_bluff_range_preflop:
                    if random.random() < self.semi_bluff_probability:
                        return ActionSpace.RAISE_POT
                return ActionSpace.FOLD

            # VS BTN
            elif open_raiser_position == pos.BTN:
                if hand in ranges['22-JJ'] or hand in ranges[
                    'AT-AQ'] or hand in ranges['A9s'] or hand in ranges[
                    'KT+'] or hand in ranges['QT+'] or hand in ranges['JTs']:
                    return ActionSpace.CHECK_CALL
                elif hand in ranges['QQ+'] or hand in ranges['AK']:
                    return ActionSpace.RAISE_POT
                elif hand in ranges['A2s-A8s'] or hand in self.common_semi_bluff_range_preflop or hand in ranges[
                    '68s'] or hand in ranges['79s']:
                    if random.random() < self.semi_bluff_probability:
                        return ActionSpace.RAISE_POT
                return ActionSpace.FOLD

            # VS SB
            if hand in open_raising_ranges[pos.CO]:
                return ActionSpace.CHECK_CALL
            elif hand in ranges['JJ+'] or hand in ranges['AK'] or hand in ranges['AQs']:
                return ActionSpace.RAISE_POT
            elif hand in ranges['K5s-K8s'] or hand in ranges['Q5s-Q7s'] or hand in ranges['J5s-J7s'] or hand in ranges[
                'T6s'] or hand in ranges['T7s'] or hand in ranges['95s-96s']:
                if random.random() < self.semi_bluff_probability:
                    return ActionSpace.RAISE_POT
            return ActionSpace.FOLD

        else:
            raise ValueError(f"Hero Position must be in [MP, CO, BTN, SB, BB] but was {hero_position}")

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
        i_th_to_act_hero = self.ordered_positions.index(hero_position)

        if obs[cols.Round_preflop]:
            # last two raises one-hot encoded for each player
            raises = self.get_raises_preflop(obs)
            # case no previous raise
            if raises['total'] == 0:
                return self.get_preflop_openraise_or_fold(obs,
                                                          hand,
                                                          hero_position)
            # case one previous raise --> it was not hero who raised:
            if raises['total'] == 1:
                # call/ xor 3b/ALLIN  xor 3b/FOLD (semi-bluff)
                return self.vs_1_preflop_raiser(hand,
                                                hero_position,
                                                raises,
                                                btn_idx)
            # case 3bet --> Hero raised previously:
            if raises['total'] > 1:
                # case 1) no-one raised twice: get earliest aggressor and latest aggressor
                # 1a) hero open raised -- vs3bet after openraise -- hero == earliest aggressor
                # --> vs latest aggressor
                # Example: HERO-SB open raises BB 3-bets todo: vs3bet_after_OR(vs_latest_aggressor)
                # 1b) if Hero is sandwhiched --> hero called before and now faces -- 3bet after Openraise
                # --> vs latest aggressor todo: vs3bet_after_OR(earliest_vs_latest_aggressor)
                # Example: UTG open-raised -- Hero-CO calls -- SB 3Bets,...UTG CALLS --> Hero acts
                # but has to act accordingt to EP vs SB 3b
                # 1c) if latest aggressor is before hero -- hero plays vs 1 Raiser
                # Example UTG open raises MP 3bets HERO-CO has to act
                # todo: vs1_raiser(vs_utg)

                # case 2) all players that raised twice are BEFORE hero
                # Example: UTG open-raises, HERO-CO 3bets, ..., UTG-4bets, HERO has to choose ALLIN / FOLD
                # todo: vs1Raiser(vs_earliest_double_raisor)
                # case 3) all players that raised twice are AFTER HERO
                # todo: vs3bet_after_openraise(vs_latest_double_raisor)
                # case 4) hero is sandwhiched by double raisors
                # todo: vs3bet_after_openraise(earliest_vs_latest_double_raisors)
                pass
        elif obs[cols.Round_flop]:
            # for postflop play, assume preflop ranges and run monte carlo sims on
            # adjusted ranges (reconstruct range from action)
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
