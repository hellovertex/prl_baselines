import enum
import os
from dataclasses import dataclass
from typing import List, Optional

from prl.environment.Wrappers.base import ActionSpaceMinimal as Action

from prl.baselines import DATA_DIR


class Stage(enum.IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


class DataImbalanceCorrection(enum.IntEnum):
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


class ActionGenOption(enum.IntEnum):
    """take (obs,action) tuples from
    top_player and drop the rest
    top player is specified in `DatasetOptions.num_top_player`"""
    # a) only if they were showdown winner
    no_folds_top_player_wins = 0
    # b) only if they participated in showdown
    # Note that the winner (obs,action)'s are possibly dropped
    no_folds_top_player_showdowns = 1

    # -- including FOLD actions
    # c) where the top player FOLD action is recorded with randomized hand
    make_folds_from_top_players_with_randomized_hand = 2
    # todo: [Optional] consider adding FOLD range
    fold_only_hands_in_range: Optional[List] = None
    # d) where the actions of the player who lost showdown are converted to FOLD
    make_folds_from_showdown_loser = 3
    # e) where the actions of the  showdown player that is not a top player is
    # converted to FOLD
    # Note that the winner (obs,action)'s are possibly dropped
    make_folds_from_non_top_player = 4


@dataclass
class DatasetOptions:
    # 01 raw -- .txt files
    num_top_players: int

    # 02 vectorized -- .csv files
    # hand histories encoded as numerical vectors
    make_dataset_for_each_individual: Optional[bool] = None
    action_generation_options: Optional[ActionGenOption] = None

    # 03 preprocessed -- .csv files
    # We exclusively use datasets of games where top players participate.
    # Usually we further limit these games to those where a top player
    # participated in showdown because we then know their hand cards.
    # The exception is when we want to use his fold actions, since we do not know his
    # hand, we randomize the cards the player had when folding, and ignore the showdown
    # when no top player participated in it.
    target_rounds: Optional[List[Stage]] = None
    action_space: Optional[List[Action]] = None

    # 99 in memory training data
    sub_sampling_techniques: Optional[DataImbalanceCorrection] = None  # dont allow
    # multiple
    # options
    # meta
    nl: str = 'NL50'

    @property
    def dir_raw_data(self):
        raw_dir = os.path.join(DATA_DIR, '01_raw')
        subdir_00_player_or_pool = 'selected_players'
        subdir_01_nl = self.nl
        return os.path.join(*[
            raw_dir,
            subdir_00_player_or_pool,
            subdir_01_nl,
        ])

    @property
    def dir_vectorized_data(self):
        assert self.make_dataset_for_each_individual is not None
        assert self.action_generation_options is not None
        vectorized_dir = os.path.join(DATA_DIR, '02_vectorized')
        subdir_00_player_or_pool = 'per_selected_player' if \
            self.make_dataset_for_each_individual else 'player_pool'
        subdir_01_nl = self.nl
        subdir_02_fold_or_no_fold = self.action_generation_options.name.replace(
            'make_',
            '')
        # when `make_dataset_for_each_individual` is set, the individual folders
        # must be created during encoding, since we dont know the ranks a priori here
        subdir_03_top_n_players = f'Top{self.num_top_players}Players' if not \
            self.make_dataset_for_each_individual else ''
        return os.path.join(*[
            vectorized_dir,
            subdir_00_player_or_pool,
            subdir_01_nl,
            subdir_02_fold_or_no_fold,
            subdir_03_top_n_players
        ])

    @property
    def dir_preprocessed_data(self):
        assert self.make_dataset_for_each_individual is not None
        assert self.action_generation_options is not None
        preprocessed_dir = os.path.join(DATA_DIR, '03_preprocessed')
        subdir_00_player_or_pool = 'per_selected_player' if \
            self.make_dataset_for_each_individual else 'player_pool'
        subdir_01_nl = self.nl
        subdir_02_fold_or_no_fold = self.action_generation_options.name.replace(
            'make_',
            '')
        # when `make_dataset_for_each_individual` is set, the individual folders
        # must be created during encoding, since we dont know the ranks a priori here
        subdir_03_top_n_players = f'Top{self.num_top_players}Players' if not \
            self.make_dataset_for_each_individual else ''
        return os.path.join(*[
            preprocessed_dir,
            subdir_00_player_or_pool,
            subdir_01_nl,
            subdir_02_fold_or_no_fold,
            subdir_03_top_n_players
        ])

    def hand_history_has_been_downloaded_and_unzipped(self):
        dir_data_unzipped = os.path.join(DATA_DIR, *['01_raw', 'all_players', self.nl])
        if os.path.exists(dir_data_unzipped):
            return True
        return False

    def exists_raw_data_for_all_selected_players(self):
        if os.path.exists(self.dir_raw_data):
            dirs_top_players = [x[0] for x in os.walk(self.dir_raw_data)]
            for i in range(self.num_top_players):
                # if no folder with name PlayerRank00i/ exists, return False
                if not f'PlayerRank{str(i).zfill(3)}' in dirs_top_players:
                    return False
        return True
