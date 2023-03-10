import enum
import os
from dataclasses import dataclass
from typing import List, Optional, Union, Type

import click
from prl.environment.Wrappers.base import ActionSpaceMinimal, ActionSpace

from prl.baselines import DATA_DIR as DEFAULT_DATA_DIR

# Preprocessed data can contain
# - Individual Action (when training dichotomizers)
# - Action Space with single bet size (ActionSpaceMinimal)
# - Action Space with multiple bet sizes (ActionSpace)
Action = Union[
    ActionSpaceMinimal,  # Allow dichotomizers only for FOLD,CHECK,RAISE (Single bet size)
    Type[ActionSpaceMinimal],
    Type[ActionSpace]
]


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
    """
    Don't change this class unless you are willing to risk the stability of the
    universe"""
    """
    Determines which (obs,action) pairs are chosen from hand histories
    """
    # Use top-players (obs,action) pairs
    # only if they participated in showdown
    # - Note that the winner (obs,action)'s are possibly dropped
    no_folds_top_player_all_showdowns = 0
    # Use top-player (obs,action) pairs
    # only if they were showdown winner
    no_folds_top_player_only_wins = 1

    # -- including FOLD actions
    # Use top-players (obs,action) pairs
    # regardless of whether they reached showdown
    # - FOLDS are obtained from top players folds only
    # - FOLD actions are recorded with randomized hands because we don't see folded hands
    make_folds_from_top_players_with_randomized_hand = 2
    # todo: [Optional] consider adding FOLD range
    # Use showdown winners (obs,action) pairs
    # regardless of whether they are a top player or not
    # - FOLDs are obtained by replacing the showdown losers action with FOLD
    make_folds_from_showdown_loser_ignoring_rank = 3
    # Use top-players (obs,action) pairs
    # only if they participated in showdown
    # - FOLDS are obtained by replacing the actions of the player that is not a top player
    #    with FOLD,
    make_folds_from_fish = 4


@dataclass
class DatasetConfig:
    """Single Source of Truth for all data related metadata"""

    # data/01_raw -- .txt files
    num_top_players: int

    # data/02_vectorized -- .csv files
    # hand histories encoded as numerical vectors
    make_dataset_for_each_individual: Optional[bool] = None
    action_generation_option: Optional[ActionGenOption] = None
    # minimum number of showdowns required to be eligible for top player ranking
    min_showdowns: int = 5000
    hudstats_enabled: bool = True

    # data/03_preprocessed -- .csv files
    # We exclusively use datasets of games where top players participate.
    # Usually we further limit these games to those where a top player
    # participated in showdown because we then know their hand cards.
    # The exception is when we want to use his fold actions, since we do not know his
    # hand, we randomize the cards the player had when folding, and ignore the showdown
    # when no top player participated in it.
    target_rounds: Optional[List[Stage]] = None
    # todo: consider removing List as we currently only support single values
    action_space: Optional[List[Action]] = None

    # 99 in memory training data
    sub_sampling_technique: Optional[
        DataImbalanceCorrection] = None  # dont allow
    # multiple
    # options
    # meta
    nl: str = 'NL50'

    """Google drive id of a .zip file containing hand histories. "
    "For small example, use 18GE6Xw4K1XE2PNiXSyh762mJ5ZCRl2SO"
    "For complete database (VERY LARGE), use "
    "18kkgEM2CYF_Tl4Dn8oro6tUgqDfr9IAN"
    "The id can be obtained from the google drive download-link url."
    "The runner will try to download the data from gdrive and proceed "
    "with unzipping."""
    from_gdrive_id: Optional[str] = "18GE6Xw4K1XE2PNiXSyh762mJ5ZCRl2SO"
    DATA_DIR: Optional[str] = None
    # Seed used to generate dataset and datasplit using
    # torch.Generator().manual_seed(seed)
    seed: int = 42

    @property
    def dir_raw_data_all_players(self):
        DATA_DIR = DEFAULT_DATA_DIR if not self.DATA_DIR else self.DATA_DIR
        raw_dir = os.path.join(DATA_DIR, '01_raw')
        # data/00_tmp and data/01_raw/all_players are irrelevant for callers
        return os.path.join(*[
            raw_dir,
            self.nl,
            'all_players',
        ])

    @property
    def dir_raw_data_top_players(self):
        DATA_DIR = DEFAULT_DATA_DIR if not self.DATA_DIR else self.DATA_DIR
        raw_dir = os.path.join(DATA_DIR, '01_raw')
        # data/00_tmp and data/01_raw/all_players are irrelevant for callers
        subdir_00_nl = self.nl
        subdir_01_player_or_pool = f'selected_players_n_showdowns={self.min_showdowns}'
        return os.path.join(*[
            raw_dir,
            subdir_00_nl,
            subdir_01_player_or_pool,
        ])

    @property
    def dir_vectorized_data(self):
        assert self.make_dataset_for_each_individual is not None
        assert self.action_generation_option is not None
        DATA_DIR = DEFAULT_DATA_DIR if not self.DATA_DIR else self.DATA_DIR
        vectorized_dir = os.path.join(DATA_DIR, '02_vectorized')
        subdir_00_nl = self.nl
        subdir_01_player_or_pool = 'per_selected_player' if \
            self.make_dataset_for_each_individual else 'player_pool'
        subdir_02_fold_or_no_fold = self.action_generation_option.name.replace(
            'make_',
            '')
        # when `make_dataset_for_each_individual` is set, the individual folders
        # must be created during encoding, since we dont know the ranks a priori here
        subdir_03_top_n_players = f'Top{self.num_top_players}Players_' \
                                  f'n_showdowns={self.min_showdowns}' if not \
            self.make_dataset_for_each_individual else ''
        subdir_04_hudstats_toggled = f'with_hudstats' if self.hudstats_enabled else ''
        return os.path.join(*[
            vectorized_dir,
            subdir_00_nl,
            subdir_01_player_or_pool,
            subdir_02_fold_or_no_fold,
            subdir_03_top_n_players,
            subdir_04_hudstats_toggled
        ])

    @property
    def dir_preprocessed_data(self):
        assert self.make_dataset_for_each_individual is not None
        assert self.action_generation_option is not None
        DATA_DIR = DEFAULT_DATA_DIR if not self.DATA_DIR else self.DATA_DIR
        preprocessed_dir = os.path.join(DATA_DIR, '03_preprocessed')
        subdir_01_player_or_pool = 'per_selected_player' if \
            self.make_dataset_for_each_individual else 'player_pool'
        subdir_00_nl = self.nl
        subdir_02_fold_or_no_fold = self.action_generation_option.name.replace(
            'make_',
            '')
        # when `make_dataset_for_each_individual` is set, the individual folders
        # must be created during encoding, since we dont know the ranks a priori here
        subdir_03_top_n_players = f'Top{self.num_top_players}Players_' \
                                  f'n_showdowns={self.min_showdowns}' if not \
            self.make_dataset_for_each_individual else ''
        subdir_04_hudstats_toggled = f'with_hudstats' if self.hudstats_enabled else ''
        subdir_05_rounds = self.target_rounds_to_str()
        subdir_06_actions = self.actions_to_str()
        return os.path.join(*[
            preprocessed_dir,
            subdir_00_nl,
            subdir_01_player_or_pool,
            subdir_02_fold_or_no_fold,
            subdir_03_top_n_players,
            subdir_04_hudstats_toggled,
            subdir_05_rounds,
            subdir_06_actions
        ])

    @property
    def dir_eval_data(self):
        DATA_DIR = DEFAULT_DATA_DIR if not self.DATA_DIR else self.DATA_DIR
        eval_dir = '04_eval'
        return os.path.join(*[
            DATA_DIR,
            eval_dir,
            self.nl
        ])

    @property
    def dir_player_summaries(self):
        DATA_DIR = DEFAULT_DATA_DIR if not self.DATA_DIR else self.DATA_DIR
        summary_dir = '99_summary'
        return os.path.join(*[
            DATA_DIR,
            summary_dir,
            self.nl,
            'player_summaries'
        ])

    @property
    def file_top_n_players_min_showdowns(self):
        """Path to .txt file containing python dictionary with Top N players and their
                serialized `PlayerSelection` data."""
        DATA_DIR = DEFAULT_DATA_DIR if not self.DATA_DIR else self.DATA_DIR
        raw_dir = os.path.join(DATA_DIR, '01_raw')
        return os.path.join(raw_dir,
                            self.nl,
                            f'top_{self.num_top_players}_players_min_showdowns={self.min_showdowns}.txt')

    def target_rounds_to_str(self):
        result = 'target_rounds='
        for stage in self.target_rounds:
            result += stage.name[0]  # use first letter for shorthand notation
        return result

    def actions_to_str(self):
        result = 'actions='
        for action in self.action_space:
            if action is ActionSpaceMinimal:
                return result + 'ActionSpaceMinimal'
            elif action is ActionSpace:
                return result + 'ActionSpace'
            else:
                assert isinstance(action,
                                  ActionSpaceMinimal), \
                    "Allow dichotomizers only for FOLD,CHECK,RAISE (Single bet size)"
                result += action.name
        return result

    def hand_history_has_been_downloaded_and_unzipped(self):
        DATA_DIR = DEFAULT_DATA_DIR if not self.DATA_DIR else self.DATA_DIR
        dir_data_unzipped = os.path.join(DATA_DIR, *['01_raw', self.nl, 'all_players'])
        if os.path.exists(dir_data_unzipped):
            return True
        return False

    def exists_raw_data_for_all_selected_players(self):
        if os.path.exists(self.dir_raw_data_top_players):
            dirs_top_players = [x[0] for x in os.walk(self.dir_raw_data_top_players)]
            for i in range(self.num_top_players):
                # if no folder with name PlayerRank00i/ exists, return False
                if not f'PlayerRank{str(i).zfill(3)}' in dirs_top_players:
                    return False
            return True
        return False

    def exists_vectorized_data_for_all_selected_players(self):
        # Traverse for n selected players, if each has their own directory
        if self.make_dataset_for_each_individual:
            if os.path.exists(self.dir_vectorized_data):
                dirs_top_players = [x[0] for x in os.walk(self.dir_vectorized_data)]
                for i in range(self.num_top_players):
                    # if no folder with name PlayerRank00i/ exists, return False
                    if not f'PlayerRank{str(i).zfill(3)}' in dirs_top_players:
                        return False
                return True
        # Otherwise simply return whether player pool vectorized data exists
        return os.path.exists(self.dir_vectorized_data)

    def exists_player_summary_data_for_all_selected_players(self):
        # Traverse for n selected players, if each has their own directory
        if os.path.exists(self.dir_player_summaries):
            dirs_top_players = [x[0] for x in os.walk(self.dir_player_summaries)]
            for i in range(self.num_top_players):
                # if no folder with name PlayerRank00i/ exists, return False
                if not f'PlayerRank{str(i).zfill(3)}' in dirs_top_players:
                    return False
        # Otherwise simply return whether player pool vectorized data exists
        return os.path.exists(self.dir_player_summaries)


arg_num_top_players = click.option("--num_top_players", default=20,
                                   type=int,
                                   help="How many top players hand histories should be used to generate the "
                                        "data.")
arg_nl = click.option("--nl",
                      default='NL50',
                      type=str,
                      help="Which stakes the hand history belongs to."
                           "Determines the data directory.")

arg_make_dataset_for_each_individual = click.option("--make_dataset_for_each_individual",
                                                    default=False,
                                                    type=bool,
                                                    help="If True, creates a designated directory per player for "
                                                         "training data. Defaults to False.")
arg_from_gdrive_id = click.option("--from_gdrive_id",
                                  default="18GE6Xw4K1XE2PNiXSyh762mJ5ZCRl2SO",
                                  type=str,
                                  help="Google drive id of a .zip file containing hand histories. "
                                       "For small example, use 18GE6Xw4K1XE2PNiXSyh762mJ5ZCRl2SO"
                                       "For complete database (VERY LARGE), use "
                                       "18kkgEM2CYF_Tl4Dn8oro6tUgqDfr9IAN"
                                       "The id can be obtained from the google drive download-link url."
                                       "The runner will try to download the data from gdrive and proceed "
                                       "with unzipping.")
arg_action_generation_option = click.option("--action_generation_option",
                                            # default=ActionGenOption.make_folds_from_top_players_with_randomized_hand.value,
                                            default=ActionGenOption.make_folds_from_top_players_with_randomized_hand.value,
                                            type=int,
                                            help="Possible Values are \n"
                                                 "0: no_folds_top_player_all_showdowns\n"
                                                 "1: no_folds_top_player_only_wins\n"
                                                 "2: make_folds_from_top_players_with_randomized_hand\n"
                                                 "3: make_folds_from_showdown_loser_ignoring_rank\n"
                                                 "4: make_folds_from_fish\n"
                                                 "See `ActionGenOption`. ")
arg_use_multiprocessing = click.option("--use_multiprocessing",
                                       default=True,
                                       type=bool,
                                       help="Whether to parallelize encoding of files per TopRanked Player. "
                                            "Defaults to True. If turned off, data generation can be VERY slow (days).")
arg_min_showdowns = click.option("--min_showdowns",
                                 default=5000,
                                 type=int,
                                 help="Minimum number of showdowns required to be eligible for top player "
                                      "ranking. Default is 5 for debugging. 5000 is recommended for real "
                                      "data.")
arg_target_rounds = click.option("--target_rounds",
                                 multiple=True,
                                 default=[Stage.PREFLOP.value],
                                 # Stage.FLOP.value,
                                 # Stage.TURN.value,
                                 # Stage.RIVER.value],
                                 type=int,
                                 help="Preprocessing will reduce data to the rounds specified. Possible values: "
                                      "Stage.PREFLOP.value: 0\nStage.FLOP.value: 1"
                                      "\nStage.TURN.value: 2\nStage.RIVER.value: 3\n"
                                      "Defaults to [FLOP,TURN,RIVER] rounds.")
arg_action_space = click.option("--action_space",
                                default="ActionSpaceMinimal",
                                type=click.Choice(["ActionSpace",
                                                   "ActionSpaceMinimal",
                                                   "FOLD",
                                                   "CHECK_CALL",
                                                   "RAISE"], case_sensitive=False),
                                help="Pick either single Action in [FOLD, CHECK_CALL, RAISE] "
                                     "if you want to train a dichotomizer or ActionSpaceMinimal "
                                     "to train using all three or ActionSpace to train on "
                                     "prl.environment.Wrappers.base.ActionSpace that has multiple bet sizes.")
arg_sub_sampling_technique = click.option("--sub_sampling_technique",
                                          default=
                                          DataImbalanceCorrection.
                                          dont_resample_but_use_label_weights_in_optimizer.value,
                                          type=int,
                                          help="Possible Values are \n"
                                               "0: dont_resample_but_use_label_weights_in_optimizer\n"
                                               "1: resample_uniform_minimal\n"
                                               "2: resample_uniform_extended\n"
                                               "3: resample_raises__to_max_num_raise\n"
                                               "See `DataImbalanceCorrection`. ")
arg_seed_dataset = click.option("--seed",
                                default=42,
                                type=int,
                                help="Seed used to generate dataset and datasplit using"
                                     "torch.Generator().manual_seed(seed)")
arg_hudstats = click.option("--hudstats",
                            default=False,
                            type=bool,
                            help="Extends observations by player statistics and hand "
                                 "strength")
