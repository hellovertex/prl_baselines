import enum
from random import randint

from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env

from prl.baselines.evaluation.v2.eval_agent import EvalAgentBase


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


turnorder = [3, 4, 5, 0, 1, 2]


def main():
    eval_agent = EvalAgentBase('hero')
    env = init_wrapped_env(AugmentObservationWrapper,
                           stack_sizes=[20000 for _ in range(6)],
                           agent_observation_mode=1,
                           blinds=(25, 50),  # = [25, 50]
                           multiply_by=100,
                           scale_rewards=True,
                           disable_info=False)
    n_samples = 1000
    for pos in turnorder:
        i = 0
        while i < n_samples:
            obs = env.reset()
            next_player = env.env.current_player.seat_id
            if next_player != pos:
                a = randint(0, 2)
                obs, _, _, _ = env.step(a)
            else:
                action = eval_agent.act(obs)
                # update ranges for position
                i += 1


if __name__ == '__main__':
    main()
