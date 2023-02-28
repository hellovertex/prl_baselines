import os
from dataclasses import dataclass

from prl.environment.Wrappers.base import ActionSpaceMinimal as Action
from prl.environment.steinberger.PokerRL import Poker

from prl.baselines.supervised_learning.v2.datasets.dataset_options import \
    DatasetOptions, DataImbalanceCorrection, ActionGenOption

from prl.baselines import DATA_DIR


@dataclass
class TrainingOptions:
    pass


def make_dataset_from_scratch(dataset_options: DatasetOptions,
                              use_multiprocessing=False) -> bool:
    opt = dataset_options
    # run main_raw(dataset_options)

    n_requested_top_players = opt.num_top_players

    # run main_vectorized(dataset_options)

    # run main_preprocess(dataset_options)
    class Preprocessor:
        @staticmethod
        def run(dataset_options: DatasetOptions):
            pass


    preprocessor = Preprocessor()
    preprocessor.run(dataset_options)
    return True


if __name__ == '__main__':
    dataset_options = DatasetOptions(
        # -- raw
        num_top_players=17,
        # -- vectorized
        make_dataset_for_each_individual=False,
        action_space=[Action.FOLD, Action.CHECK_CALL, Action.RAISE],
        # -- preprocessed
        target_rounds=[Poker.FLOP, Poker.TURN, Poker.RIVER],
        sub_sampling_techniques=DataImbalanceCorrection
        .dont_resample_but_use_label_weights_in_optimizer,
        action_generation_options=ActionGenOption
        .make_folds_from_top_players_with_randomized_hand
    )

    make_dataset_from_scratch(dataset_options)



# end to end data generation via
# make_dataset_from_scratch(dataset_options)  # todo parallelization_options

# run main_raw
# run main_vectorized
# run main_preprocessed
# el fin

# todo: implement the stubs, consider creating branch
# main_raw
# check if 01_raw/all_players/NL50 exist
# if not, gdrive id and then unzip recursively to opt.get_raw_dir
# check if 01_raw/selected_players/NL50 exist up to rank specified by num_top_players
# if not make folders for each missing rank using hsmithy extractor


# main_vectorized
# 2a) data is present -- continue
# 2b) 02_vec: data is not present -- make dataset using `target_rounds`
# `actions_to_predict`

# main_preprocessed
# 3a) data is present -- continue
# 3b) 03_preprocessed: preprocess using sub_sampling_techniques

# end to end training using `opt`
# 1. assert all necessary training data is available
# 2. Make InMemoryDataset using opt.sub_sampling_techniques
# training(dataset_options, training_options)

# todo: implement new Preprocessor(dataset_options) with dir_preprocessed + suffix
# todo: implement new training methods
# todo: implement MC on hand range
