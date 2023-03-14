import glob
import os
from pathlib import Path

import pandas as pd
import pytest
from prl.environment.Wrappers.base import ActionSpace, ActionSpaceMinimal

from prl.baselines.supervised_learning.v2.datasets.dataset_config import ActionGenOption, \
    DatasetConfig, Stage
from prl.baselines.supervised_learning.v2.datasets.preprocessed_data import \
    make_preprocessed_data_if_not_exists_already, PreprocessedData

vectorized_observation_shape = 569


# load vectorized data
# from default directory
@pytest.fixture
def dataset_config():
    num_top_players = 20
    nl = 'NL50'
    from_gdrive_id = '18GE6Xw4K1XE2PNiXSyh762mJ5ZCRl2SO'
    make_dataset_for_each_individual = False
    action_generation_option = ActionGenOption. \
        make_folds_from_top_players_with_randomized_hand
    min_showdowns = 10
    data_dir = os.path.join(Path(__file__).parent, 'data')
    dataset_config = DatasetConfig(num_top_players=num_top_players,
                                   make_dataset_for_each_individual=make_dataset_for_each_individual,
                                   action_generation_option=action_generation_option,
                                   min_showdowns=min_showdowns,
                                   nl=nl,
                                   from_gdrive_id=from_gdrive_id,
                                   target_rounds=[
                                       Stage.FLOP,
                                       Stage.TURN,
                                       Stage.RIVER],
                                   action_space=[ActionSpaceMinimal],
                                   DATA_DIR=data_dir)
    return dataset_config


@pytest.fixture
def csv_files(dataset_config):
    make_preprocessed_data_if_not_exists_already(dataset_config,
                                                 use_multiprocessing=True)
    return glob.glob(dataset_config.dir_preprocessed_data + '/**/*.csv.bz2',
                     recursive=True)


def test_preprocessing_is_skipped_if_all_preprocessed_data_exists_already(dataset_config):
    preprocessed_data = PreprocessedData(dataset_config)
    make_preprocessed_data_if_not_exists_already(dataset_config,
                                                 use_multiprocessing=True)
    # `csv_files` fixture generates preprocessed data via
    # `make_preprocessed_data_if_not_exists_already`
    assert preprocessed_data.all_vectorized_data_has_been_preprocessed_before()


def assert_all_actions_present_and_no_other_actions_as_specified(df, action_space):
    assert not (df['label'] > max(action_space)).any()
    assert not (df['label'] < min(action_space)).any()
    # todo: remove the following -- it does not always hold
    # for a in list(action_space):
    #     assert (df['label'] == a.value).any()


def test_preprocessor_action_clipping(dataset_config, csv_files):
    act = dataset_config.action_space[0]
    for file in csv_files:
        df = pd.read_csv(file,
                         sep=',',
                         dtype='float32',
                         encoding='cp1252',
                         compression='bz2')
        assert df['label'].max
        if act is ActionSpace:
            assert_all_actions_present_and_no_other_actions_as_specified(df, act)
        elif act is ActionSpaceMinimal:
            assert_all_actions_present_and_no_other_actions_as_specified(df, act)
        elif isinstance(act, ActionSpaceMinimal):
            assert_all_actions_present_and_no_other_actions_as_specified(df, act)


def test_preprocessor_target_round_clipping(dataset_config, csv_files):
    for file in csv_files:
        df = pd.read_csv(file,
                         sep=',',
                         dtype='float32',
                         encoding='cp1252',
                         compression='bz2')
        for stage in dataset_config.target_rounds:
            name = 'round_' + stage.name.casefold()
            assert (df[name] == 1).any()
        for stage in list(Stage):
            if stage not in dataset_config.target_rounds:
                name = 'round_' + stage.name.casefold()
                assert not (df[name] == 1).any()


def test_preprocessor_not_changing_vectorized_observation_1(csv_files):
    for file in csv_files:
        df = pd.read_csv(file,
                         sep=',',
                         dtype='float32',
                         encoding='cp1252',
                         compression='bz2')
        for r in df.to_numpy():
            # each column can not be much larger than one and most of them are zero,
            # so if the sum along all dimensions is greater than the length of
            # the vector, the data has changed after vectorization
            assert sum(r) < vectorized_observation_shape


def test_preprocessor_works_per_selected_player(dataset_config):
    # per selected player == with hud stats
    dataset_config.make_dataset_for_each_individual = True
    dataset_config.hudstats_enabled = False
    make_preprocessed_data_if_not_exists_already(dataset_config, True)
    assert PreprocessedData(
        dataset_config).all_vectorized_data_has_been_preprocessed_before()
    # per selected player == no hud stats
    dataset_config.make_dataset_for_each_individual = True
    dataset_config.hudstats_enabled = True
    make_preprocessed_data_if_not_exists_already(dataset_config, True)
    assert PreprocessedData(
        dataset_config).all_vectorized_data_has_been_preprocessed_before()
