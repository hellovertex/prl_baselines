# preflop equities
# flop pot odds
# assume ranges (consider making them stochastic) for post flop MC analysis
import numpy as np
from prl.environment.Wrappers.aoh import Positions6Max
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as cols

from prl.baselines.evaluation.utils import get_reset_config, pretty_print
from prl.baselines.examples.examples_tianshou_env import make_default_tianshou_env
from prl.environment.Wrappers.vectorizer import Vectorizer

hand0 = []


class RuleBasedAgent:
    def __init__(self, num_players):
        # assume that number of players does not change during the game
        # this assumption is valid, because we refill each player stack
        # after each round, such that the number of players never decreases
        self.num_players = num_players
        pos = Positions6Max
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

    def act(self, obs: np.ndarray, legal_moves):
        print('YEAY')
        c0 = obs[cols.First_player_card_0_rank_0:cols.First_player_card_0_suit_3 + 1]
        c1 = obs[cols.First_player_card_1_rank_0:cols.First_player_card_1_suit_3 + 1]
        r0, s0 = np.where(c0 == 1)[0]
        r1, s1 = np.where(c1 == 1)[0]
        max_r = max(r0, r1)
        min_r = min(r0, r1)
        hand = (max_r, min_r) if s0 == s1 else (min_r, max_r)
        positions = list(self.positions)
        btn_idx = np.where(obs[cols.Btn_idx_is_0:cols.Btn_idx_is_5 + 1] == 1)[0]
        # fuer jeden Spieler die letzten beiden raises one hot encoded
        raises = self.get_raises_preflop(obs)
        # map diese aktionen auf die position
        # pos_idx = (i + btn_idx) % self.num_players
        a = 1
        # case no previous raise
        if raises['total'] == 0:
            # bet/fold
            pass
        # case one previous raise:
        if raises['total'] == 1:
            # call/ xor 3b/ALLIN  xor 3b/FOLD (semi-bluff)
            pass
        # case 3bet:
        if raises['total'] > 1:
            # 4b/ALLIN or Call but we dont semi-bluff
            pass

        if obs[cols.Round_preflop]:
            # act according to preflop equity chart
            # open for 4 BB -- UTG
            # origin (0,0) at 22
            # hand Q3s at (10,1) Q3o (1,10)
            c0 = obs[cols.First_player_card_0_rank_0:cols.First_player_card_0_suit_3 + 1]
            c1 = obs[cols.First_player_card_1_rank_0:cols.First_player_card_1_suit_3 + 1]

            # if cards are suited -- higher card is column else row
            pass
        elif obs[cols.Round_flop]:
            pass
        elif obs[cols.Round_turn]:
            pass
        elif obs[cols.Round_river]:
            pass
        else:
            raise ValueError("Observation does not contain stage-bit. "
                             "It either is a terminal observation or not initialized")
        return 0


if __name__ == '__main__':
    num_players = 3
    verbose = True
    hidden_dims = [256]
    starting_stack = 20000
    stack_sizes = [starting_stack for _ in range(num_players)]
    agent_names = [f'p{i}' for i in range(num_players)]
    # rainbow_config = get_rainbow_config(default_rainbow_params)
    # RainbowPolicy(**rainbow_config).load_state_dict...
    # env = get_default_env(num_players, starting_stack)
    env = make_default_tianshou_env(mc_model_ckpt_path=None,  # dont use mc
                                    agents=agent_names,
                                    num_players=len(agent_names))
    agents = [RuleBasedAgent(num_players) for _ in range(num_players)]
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
