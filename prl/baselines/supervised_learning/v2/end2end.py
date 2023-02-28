# specify parameters
# available options:
from prl.environment.Wrappers.base import ActionSpaceMinimal as Action
from prl.environment.steinberger.PokerRL import Poker

num_top_players = 17
make_dataset_for_each_individual = False
target_rounds: [Poker.FLOP, Poker.TURN, Poker.RIVER]
actions_to_predict = [Action.FOLD, Action.CHECK_CALL, Action.RAISE]

# script should
# look for dataset in designated folder
# 1) txt databases
# a) data is present -- continue
# b) data is not present -- make dataset using `num_top_players` and `make_dataset_for_each_individual`
# 2) vectorized databases
# a) data is present -- continue
# b)
