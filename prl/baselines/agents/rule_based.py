# preflop equities
# flop pot odds
# assume ranges (consider making them stochastic) for post flop MC analysis
import random
from typing import List

import numpy as np
from prl.environment.Wrappers.aoh import Positions6Max as pos
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as cols
from prl.environment.Wrappers.base import ActionSpace

from prl.baselines.agents.hand_ranges import open_raising_ranges, vs_1_raiser_3b_and_fold, vs_1_raiser_call, \
    vs_1_raiser_3b_and_allin, vs_3bet_after_openraise_call, vs_3bet_after_openraise_4b_and_allin
from prl.baselines.evaluation.utils import get_reset_config, pretty_print
from prl.baselines.examples.examples_tianshou_env import make_default_tianshou_env


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
        self.open_raising_sizes = (4, 3, 2.5, 2, 2, 4)
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

    @staticmethod
    def vs_1_raiser_pf(hand, defender, aggressor, is_first_betting_round):
        if hand in vs_1_raiser_3b_and_fold[defender][aggressor]:
            if is_first_betting_round:
                return ActionSpace.RAISE_POT
            else:
                return ActionSpace.FOLD
        elif hand in vs_1_raiser_3b_and_allin[defender][aggressor]:
            if is_first_betting_round:
                return ActionSpace.RAISE_POT
            else:
                return ActionSpace.RAISE_ALL_IN
        elif hand in vs_1_raiser_call[defender][aggressor]:
            return ActionSpace.CHECK_CALL
        return ActionSpace.FOLD

    @staticmethod
    def vs_3bet_after_openraise(hand, defender, aggressor, is_first_betting_round):
        if hand in vs_3bet_after_openraise_call[defender][aggressor]:
            return ActionSpace.CHECK_CALL
        elif hand in vs_3bet_after_openraise_4b_and_allin[defender][aggressor]:
            if is_first_betting_round:
                return ActionSpace.RAISE_POT
            else:
                return ActionSpace.RAISE_ALL_IN
        return ActionSpace.FOLD

    @staticmethod
    def is_first_betting_round(raises):
        if sum(raises[0]) > 1 or sum(raises[1]) > 1 or sum(raises[2]) > 1 or sum(raises[3]) > 1 or sum(
                raises[4]) > 1 or sum(raises[5]) > 1:
            return False
        return True

    def get_players_who_raised_twice_preflop(self, raises) -> List[int]:
        result = []
        if sum(raises[0]) > 1:
            result.append(0)
        if sum(raises[1]) > 1:
            result.append(1)
        if sum(raises[2]) > 1:
            result.append(2)
        if sum(raises[3]) > 1:
            result.append(3)
        if sum(raises[4]) > 1:
            result.append(4)
        if sum(raises[5]) > 1:
            result.append(5)
        return result

    def act(self, obs: np.ndarray, legal_moves):
        print('YEAY')
        c0 = obs[cols.First_player_card_0_rank_0:cols.First_player_card_0_suit_3 + 1]
        c1 = obs[cols.First_player_card_1_rank_0:cols.First_player_card_1_suit_3 + 1]
        r0, s0 = np.where(c0 == 1)[0]
        r1, s1 = np.where(c1 == 1)[0]
        max_r = max(r0, r1)
        min_r = min(r0, r1)
        hand = (max_r, min_r) if s0 == s1 else (min_r, max_r)
        btn_idx = np.where(obs[cols.Btn_idx_is_0:cols.Btn_idx_is_5 + 1] == 1)[0][0]
        if obs[cols.Round_preflop]:
            # last two raises one-hot encoded for each player
            raises = self.get_raises_preflop(obs)
            hero_position: pos = self.positions[-btn_idx % self.num_players]

            # true if nobody raised twice
            is_first_betting_round = self.is_first_betting_round(raises)

            # case nobody raised previously
            if raises['total'] == 0:
                return self.get_preflop_openraise_or_fold(obs,
                                                          hand,
                                                          hero_position)
            # index relative to button
            aggressor1 = min(raises['who_raised'])
            aggressor2 = max(raises['who_raised'])

            # Positions6Max names
            aggressor1_position: pos = self.positions[(aggressor1 - btn_idx) % self.num_players]
            aggressor2_position: pos = self.positions[(aggressor2 - btn_idx) % self.num_players]

            # turn order
            hero_index: int = self.ordered_positions.index(hero_position)
            agg1_index: int = self.ordered_positions.index(aggressor1_position)
            agg2_index: int = self.ordered_positions.index(aggressor2_position)
            latest_aggressor: int = max(agg1_index, agg2_index)
            earliest_aggressor: int = min(agg1_index, agg2_index)

            # case one previous raise --> it was not hero who raised:
            if raises['total'] == 1:
                # call/ xor 3b/ALLIN  xor 3b/FOLD (semi-bluff)
                # defender = hero
                assert hero_index > agg1_index
                assert agg1_index == agg2_index
                assert is_first_betting_round
                return self.vs_1_raiser_pf(hand,
                                           defender=hero_position,
                                           aggressor=aggressor1_position,
                                           is_first_betting_round=is_first_betting_round)

            # case 3bet --> Hero raised previously:
            if raises['total'] > 1:
                # is empty when nobody raised twice
                players_that_raised_twice = self.get_players_who_raised_twice_preflop(raises)

                # case 1) no-one raised twice
                if is_first_betting_round:
                    assert not players_that_raised_twice
                    if latest_aggressor > hero_index:
                        # 1a) hero open raised -- faces 3bet after openraise -- hero == earliest aggressor
                        # Example: HERO-SB open raises BB 3-bets
                        if earliest_aggressor == hero_index:
                            aggressor = aggressor1_position if latest_aggressor == agg1_index else aggressor2_position
                            return self.vs_3bet_after_openraise(hand,
                                                                defender=hero_position,
                                                                aggressor=aggressor,
                                                                is_first_betting_round=True)
                        # 1b) if Hero is sandwhiched --> hero called before and now faces -- 3bet after Openraise
                        # Example: UTG open-raised -- Hero-CO calls -- SB 3Bets,...UTG CALLS --> Hero acts
                        elif earliest_aggressor < hero_index < latest_aggressor:
                            defender = aggressor1_position if earliest_aggressor == agg1_index else aggressor2_position
                            aggressor = aggressor1_position if latest_aggressor == agg1_index else aggressor2_position
                            return self.vs_3bet_after_openraise(hand,
                                                                defender=defender,  # hero has to use ranges of defender
                                                                aggressor=aggressor,
                                                                is_first_betting_round=True)
                    else:
                        # 1c) if latest aggressor is before hero -- hero plays vs 1 Raiser (possibly 3bet pot already)
                        # Example UTG open raises MP 3bets HERO-CO has to act
                        assert latest_aggressor < hero_index
                        defender = hero_position
                        aggressor = aggressor1_position if earliest_aggressor == agg1_index else aggressor2_position
                        return self.vs_1_raiser_pf(hand,
                                                   defender=defender,
                                                   aggressor=aggressor,
                                                   is_first_betting_round=True)
                # case 2) at least one player raised twice
                else:
                    raisors = players_that_raised_twice
                    position_min = self.positions[(min(raisors) - btn_idx) % self.num_players]
                    index_min = self.ordered_positions.index(position_min)
                    position_max = self.positions[(max(raisors) - btn_idx) % self.num_players]
                    index_max = self.ordered_positions.index(position_max)
                    earliest = position_max if index_max < index_min else position_min
                    latest = position_max if index_max > index_min else position_min
                    # case 2a) all players that raised twice are BEFORE hero
                    # Example: UTG open-raises, HERO-CO 3bets, ..., UTG-4bets, HERO has to choose ALLIN / FOLD
                    if max(index_min, index_max) < hero_index:
                        return self.vs_1_raiser_pf(hand,
                                                   defender=hero_position,
                                                   aggressor=earliest,
                                                   is_first_betting_round=False)
                    # case 2b) hero is sandwhiched by double raisors
                    # Example: UTG open raises Hero calls SB 3 bets
                    # continued:  UTG 4 bets Hero calls SB 5 bets UTG calls -- Hero has to act
                    elif min(index_min, index_max) < hero_index < max(index_min, index_max):
                        return self.vs_3bet_after_openraise(hand,
                                                            defender=earliest,  # hero must defend using UTG ranges
                                                            aggressor=latest,
                                                            is_first_betting_round=False)
                    # case 2c) all players that raised twice are AFTER HERO
                    # Example: HERO BTN limps SB Openraises BB 3bets
                    # continued: Hero calls SB 4bets BB 5 bets
                    else:
                        return self.vs_3bet_after_openraise(hand,
                                                            defender=earliest,  # hero must defend vs BB using SB ranges
                                                            aggressor=latest,
                                                            is_first_betting_round=False)
        elif obs[cols.Round_flop]:
            # for postflop play, assume preflop ranges and run monte carlo sims on
            # adjusted ranges (reconstruct range from action)
            return ActionSpace.FOLD
        elif obs[cols.Round_turn]:
            return ActionSpace.FOLD
        elif obs[cols.Round_river]:
            return ActionSpace.FOLD
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
    # player_hands = ['[Jh Jc]', '[4h 6s]', '[As 5s]']
    player_hands = ['[Ah Ac]', '[Kh Ks]', '[As Ad]']
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
            pretty_print(i, obs, action, env.env.env.env_wrapped)
            obs_dict, cum_reward, terminated, truncated, info = env.step(action)
            rews = cum_reward
            agent_id = obs_dict['agent_id']
            print(f'AGENT_ID', agent_id)
            obs = obs_dict['obs']
            print(f'GOT REWARD {cum_reward}')
            if terminated:
                break
