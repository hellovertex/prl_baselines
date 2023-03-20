import enum
from random import randint

import numpy as np
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env
from prl.environment.Wrappers.vectorizer import AgentObservationType
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as cols
from prl.baselines.evaluation.v2.eval_agent import EvalAgentBase
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
    return (max_r, min_r) if s0 == s1 else (min_r, max_r)


def update_ranges(plays, folds, action, position, obs):
    # if action is fold, check that action is integer and not tensor, ndarray, tuple etc
    # increment card counter -- don't compute ratios, just use absolute numbers here
    ci, cj = get_card_from_obs(obs)
    if action == 0:
        # todo: make cure ci, cj are in correct order
        folds[position][ci][cj] += 1
    else:
        plays[position][ci][cj] += 1
    return plays, folds


turnorder = [3, 4, 5, 0, 1, 2]


def main():
    eval_agent = EvalAgentBase('hero')
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
        while i < n_samples:
            obs = env.reset(None)
            next_player = env.env.current_player.seat_id
            if next_player != pos:
                a = randint(0, 2)
                obs, _, _, _ = env.step(a)
            else:
                action = eval_agent.act(obs)
                update_ranges(ranges_played,
                              ranges_folded,
                              action,
                              pos,
                              obs)
                i += 1
    for pos in Positions6Max:
        ratios = ranges_played[pos] / ranges_folded[pos]
        plot_ranges(ratios)


if __name__ == '__main__':
    main()
