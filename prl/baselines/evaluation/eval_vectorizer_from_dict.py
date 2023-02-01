from typing import Tuple

import numpy as np
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as fts
from prl.environment.Wrappers.base import ActionSpace
from prl.environment.steinberger.PokerRL import NoLimitHoldem, Poker

from prl.baselines.evaluation.analyzer import PlayerAnalyzer
from prl.baselines.evaluation.utils import get_reset_config
from prl.baselines.examples.examples_tianshou_env import MCAgent
from prl.baselines.supervised_learning.data_acquisition.environment_utils import make_board_cards, card_tokens, card


def parse_action(env, int_action) -> Tuple:
    """for testing only, we do not care about raise amounts,
    as we want to verify the vectorizer works correctly"""
    if int_action in [0, 1]:
        return (int_action, -1)
    elif int_action == ActionSpace.RAISE_MIN_OR_3BB:
        return (2, 100)
    elif int_action == ActionSpace.RAISE_HALF_POT:
        return (2, 100)
    elif int_action == ActionSpace.RAISE_POT:
        return (2, 100)
    elif int_action == ActionSpace.ALL_IN:
        return (2, 100)
    else:
        raise ValueError


num_players = 3
starting_stack_size = 20000
stack_sizes = [starting_stack_size for _ in range(num_players)]
args = NoLimitHoldem.ARGS_CLS(n_seats=len(stack_sizes),
                              starting_stack_sizes_list=stack_sizes)
# return wrapped env instance
env = NoLimitHoldem(is_evaluating=True,
                    env_args=args,
                    lut_holder=NoLimitHoldem.get_lut_holder())
wrapped_env = AugmentObservationWrapper(env)
# will be used for naming feature index in training data vector
feature_names = list(wrapped_env.obs_idx_dict.keys())

ckpt = "/home/sascha/Documents/github.com/prl_baselines/data/ckpt/ckpt.pt"

agents = [
    MCAgent(ckpt, num_players),
    MCAgent(ckpt, num_players),
    MCAgent(ckpt, num_players),
]
# # '[6h Ts]' to ['6h', 'Ts']
# showdown_cards = card_tokens(final_player.cards)
# # ['6h', 'Ts'] to [[5,3], [5,0]]
# hand = [card(token) for token in showdown_cards]
board = '[6h Ts Td 9c Jc]'
player_hands = ['[6s 6d]', '[9s 9d]', '[Jd Js]']
state_dict = get_reset_config(player_hands, board)
unzipped_dir = "/home/sascha/Documents/github.com/prl_baselines/data/01_raw/0.25-0.50/player_data_test"

ckpt_path = "/home/sascha/Documents/github.com/prl_baselines/data/ckpt/ckpt.pt"
baseline = MCAgent(ckpt_path, num_players)

obs, rew, done, info = wrapped_env.reset({'deck_state_dict': state_dict})
assert len(agents) == num_players == len(stack_sizes)
i = 0
while True:
    legal_moves = wrapped_env.get_legal_moves_extended()
    action = agents[i].compute_action(obs, legal_moves)
    # action = parse_action(wrapped_env, action)
    pred = baseline.compute_action(obs, legal_moves)
    obs, rew, done, info = wrapped_env.step(action)

    if not done:
        i = (i + 1) % num_players
    # make sure the card feature columns mathc the cards of the resp players
    # make sure the current bet feature columns mathc the current bet of the resp players
    # stack etc make sure they all match
    hands = []
    for s in env.seats:
        hands.append(s.hand)
    p0_first_card = obs[fts.First_player_card_0_rank_0:fts.First_player_card_1_rank_0]
    p0_second_card = obs[fts.First_player_card_1_rank_0:fts.Second_player_card_0_rank_0]
    p1_first_card = obs[fts.Second_player_card_0_rank_0:fts.Second_player_card_1_rank_0]
    p1_second_card = obs[fts.Second_player_card_1_rank_0:fts.Third_player_card_0_rank_0]
    r00 = hands[i][0][0]  # rank first player first card
    s00 = hands[i][0][1]
    r01 = hands[i][1][0]
    s01 = hands[i][1][1]  # suite first player second card
    """
    a0 calls after reset
    a1 observes obs1 folds
    a2 observes obs2 folds --> Game ended
    who gets obs3? a2 gets ob3 but a0 and a1 are also candidates. however they wont.
    for simplicity in these cases the transitions are cut out and only 
    the transition for a2 survives
    """
    # note after done we dont increment i, so the last remaining player gets obs
    assert p0_first_card[r00] == 1
    assert p0_first_card[13 + s00] == 1
    assert p0_second_card[r01] == 1
    assert p0_second_card[13 + s01] == 1
    if not done:
        assert sum(p1_first_card) == 0
        assert sum(p1_second_card) == 0
    else:
        assert sum(p1_first_card) == 2
        assert sum(p1_second_card) == 2
    if done:
        break

    a = 0
