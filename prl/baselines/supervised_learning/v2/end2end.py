# specify parameters
# available options:
import enum

from prl.environment.Wrappers.base import ActionSpaceMinimal as Action
from prl.environment.steinberger.PokerRL import Poker


class DataImbalanceCorrections(enum.IntEnum):
    """Dataset labels are likely imbalanced. For example the number of
    ALL_IN actions is smaller than the number of CHECK_CALL actions.
    There are several ways to counter this when doing Machine learning
    on the data:
    """
    # Leave dataset as is -- provide the optimizer with a list of
    # weights that is proportional to label-frequency. This is the
    # recommended approach, but  not all optimizers support this
    # technique.
    dont_resample_but_use_label_weights_in_optimizer = 0
    # |FOLDS| = |CHECK_CALLS| = |RAISES|
    # usually means slightly downsampling FOLDS and CHECK_CALLS
    # and slightly upsamling RAISES
    resample_uniform_minimal = 1
    # |FOLDS| = |CHECK_CALLS| = |RAISE_1| = ... = |RAISE_n|
    # usually means heavily downsampling FOLDS and CHECK_CALLS and
    # heavily upsamling RAISES, especially for larger `n` this is
    # usally not recommended, because of the additional bias
    resample_uniform_extended = 2
    # |FOLD| = |CHECK_CALLS| >> max( {|RAISE_1|,...,|RAISE_n|} ) =
    # |RAISE_1| = ... = |RAISE_n|
    # one of FOLD / CHECK_CALLS gets slightly resampled. The raise
    # actions get upsampled, such that their label-frequency is
    # uniformly distributed.
    resample_raises__to_max_num_raise = 3
    resample_raises__to_max_num_label = 4


# dataset options
# -- raw
num_top_players = 17
make_dataset_for_each_individual = False
# -- vectorized
target_rounds: [Poker.FLOP, Poker.TURN, Poker.RIVER]
actions_to_predict = [Action.FOLD, Action.CHECK_CALL, Action.RAISE]
# -- preprocessed
sub_sampling_technique = 'equal'

# script should
# look for dataset in designated folder
# 1) txt databases
# a) data is present -- continue
# b) 01_raw: data is not present -- make dataset using `num_top_players` and `make_dataset_for_each_individual`
# 2) vectorized databases
# a) data is present -- continue
# b) 02_vec: data is not present -- make dataset using `target_rounds` `actions_to_predict`
# 3) preprocessed databases
# a) data
