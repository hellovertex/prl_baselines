# specify parameters
# available options:
import enum
from typing import List

from dataclasses import dataclass
from prl.environment.Wrappers.base import ActionSpaceMinimal as Action
from prl.environment.steinberger.PokerRL import Poker


class DataImbalanceCorrections(enum.IntEnum):
    """Dataset labels are likely imbalanced. For example the number of
    ALL_IN actions is smaller than the number of CHECK_CALL actions.
    There are several ways to counter this when doing Machine learning:
    """
    # Leave dataset as is -- provide the optimizer with a list of
    # weights that is proportional to label-frequency. This is the
    # recommended approach, but  not all optimizers support this
    # technique.
    dont_resample_but_use_label_weights_in_optimizer = 0
    # |FOLDS| = |CHECK_CALLS| = |RAISES|
    # usually means slightly downsampling FOLDS and CHECK_CALLS
    # and slightly upsamling RAISES.
    # NOTE: requires actions from `prl.environment.Wrappers.base.ActionSpaceMinimal`
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
    # uniformly distributed. This results in approximate balance:
    # |FOLDS| = |CHECK_CALLS| ~= sum({|RAISE_1|,...,|RAISE_n|})
    # Recommended when label weights can not be used and having
    # an ActionSpace with multiple BET sizes.
    resample_raises__to_max_num_raise = 3


class Stage(enum.IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


@dataclass
class DatasetOptions:
    # raw -- .txt files
    num_top_players: int
    make_dataset_for_each_individual: bool
    # vectorized -- .csv files
    target_rounds: List[Stage]
    actions_to_predict: List[Action]
    # preprocessed -- .csv files
    sub_sampling_techniques: List[DataImbalanceCorrections]


@dataclass
class TrainingOptions:
    pass


# dataset options
# -- raw
num_top_players = 17
make_dataset_for_each_individual = False
# -- vectorized
target_rounds: [Poker.FLOP, Poker.TURN, Poker.RIVER]
actions_to_predict = [Action.FOLD, Action.CHECK_CALL, Action.RAISE]
# -- preprocessed
sub_sampling_techniques = [DataImbalanceCorrections.dont_resample_but_use_label_weights_in_optimizer,
                           DataImbalanceCorrections.resample_uniform_minimal]

# script should
# look for dataset in designated folder
# 1) txt databases
# 1a) data is present -- continue
# 1b) 01_raw: data is not present -- make dataset using `num_top_players` and `make_dataset_for_each_individual`
# 2) vectorized databases
# 2a) data is present -- continue
# 2b) 02_vec: data is not present -- make dataset using `target_rounds` `actions_to_predict`
# 3) preprocessed databases
# 3a) data is present -- continue
# 3b) 03_preprocessed: preprocess using sub_sampling_techniques
# 4) train
