import enum
from random import randint

import numpy as np
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env
from prl.environment.Wrappers.vectorizer import AgentObservationType
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as cols
from prl.reinforce.train_eval import RainbowConfig
from tianshou.policy import RainbowPolicy
from tqdm import tqdm

from prl.baselines.agents.tianshou_policies import get_rainbow_config
from prl.baselines.evaluation.v2.eval_agent import EvalAgentBase, \
    EvalAgentRanges, EvalAgentCall
from prl.baselines.evaluation.v2.plot_range_charts import plot_ranges


class Positions6Max(enum.IntEnum):
    """Positions as in the literature, for a table with at most 6 Players.
    BTN for Button, SB for Small Blind, etc...
    """
    BTN = 0
    SB = 1
    BB = 2
    UTG = 3  # UnderTheGun
    MP = 4  # Middle Position
    CO = 5  # CutOff


def get_card_from_obs(obs):
    c0 = obs[cols.First_player_card_0_rank_0:cols.First_player_card_0_suit_3 + 1]
    c1 = obs[cols.First_player_card_1_rank_0:cols.First_player_card_1_suit_3 + 1]
    r0, s0 = np.where(c0 == 1)[0]
    r1, s1 = np.where(c1 == 1)[0]
    max_r = max(r0, r1)
    min_r = min(r0, r1)
    return (12 - max_r, 12 - min_r) if s0 == s1 else (12 - min_r, 12 - max_r)


def update_ranges(plays, folds, action, position, obs):
    # if action is fold, check that action is integer and not tensor, ndarray, tuple etc
    # increment card counter -- don't compute ratios, just use absolute numbers here
    ci, cj = get_card_from_obs(obs)  # e.g (8,10) for QTo or (10,8) for QTs
    if action == 0:
        folds[position][ci][cj] += 1
    else:
        plays[position][ci][cj] += 1
    return plays, folds


turnorder = [3, 4, 5, 0, 1, 2]


def main():
    ckpt_save_path = '/home/sascha/Documents/github.com/prl_baselines/data/checkpoints/vs_calling_station/ckpt.pt'
    params = {'device': 'cpu',
              'load_from_ckpt': ckpt_save_path,
              'lr': 1e-6,
              'num_atoms': 51,
              'noisy_std': 0.1,
              'v_min': -6,
              'v_max': 6,
              'estimation_step': 1,
              'target_update_freq': 5000
              # training steps
              }
    rainbow_config = get_rainbow_config(params)
    rainbow = RainbowPolicy(**rainbow_config)
    eval_agent = EvalAgentRanges('hero', rainbow)
    #eval_agent = EvalAgentCall('caller')

    env = init_wrapped_env(AugmentObservationWrapper,
                           stack_sizes=[20000 for _ in range(6)],
                           agent_observation_mode=AgentObservationType.CARD_KNOWLEDGE,
                           blinds=(25, 50),  # = [25, 50]
                           multiply_by=100,
                           scale_rewards=True,
                           disable_info=False)
    n_samples = 1000
    ranges_played = {Positions6Max.UTG: np.zeros((13, 13)),
                     Positions6Max.MP: np.zeros((13, 13)),
                     Positions6Max.CO: np.zeros((13, 13)),
                     Positions6Max.BTN: np.zeros((13, 13)),
                     Positions6Max.SB: np.zeros((13, 13)),
                     Positions6Max.BB: np.zeros((13, 13))}
    ranges_folded = {Positions6Max.UTG: np.zeros((13, 13)),
                     Positions6Max.MP: np.zeros((13, 13)),
                     Positions6Max.CO: np.zeros((13, 13)),
                     Positions6Max.BTN: np.zeros((13, 13)),
                     Positions6Max.SB: np.zeros((13, 13)),
                     Positions6Max.BB: np.zeros((13, 13))}
    for pos in turnorder:
        i = 0
        obs, _, _, _ = env.reset(None)
        pbar = tqdm(total=n_samples)
        while i < n_samples:
            next_player = env.env.current_player.seat_id
            if next_player != pos:
                a = randint(0, 7)
                obs, _, _, _ = env.step(a)
            else:
                action = eval_agent.act(obs)
                update_ranges(ranges_played,
                              ranges_folded,
                              action,
                              pos,
                              obs)
                obs, _, _, _ = env.reset(None)
                i += 1
                pbar.update(1)
        pbar.close()
    ranges = {}
    for pos in Positions6Max:
        ranges[pos] = ranges_played[pos]  # / (ranges_folded[pos] + ranges_played[pos])
    plot_ranges(ranges)


if __name__ == '__main__':
    main()
