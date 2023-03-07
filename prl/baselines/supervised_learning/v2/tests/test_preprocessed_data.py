import glob
import os
from pathlib import Path

import pandas as pd
import pytest
from prl.environment.Wrappers.base import ActionSpace, ActionSpaceMinimal

from prl.baselines.supervised_learning.v2.datasets.dataset_config import ActionGenOption, \
    DatasetConfig, Stage
from prl.baselines.supervised_learning.v2.datasets.preprocessed_data import \
    make_preprocessed_data_if_not_exists_already


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
                                   target_rounds=[],
                                   action_space=[],
                                   DATA_DIR=data_dir)
    return dataset_config


@pytest.fixture
def csv_files(dataset_config):
    make_preprocessed_data_if_not_exists_already(dataset_config,
                                                 use_multiprocessing=True)
    return glob.glob(dataset_config.dir_vectorized_data + '/**/*.csv.bz2',
                     recursive=True)

def assert_all_actions_present_and_no_other_actions_as_specified(df, action_space):
    assert not df['label'] > max(action_space)
    assert not df['label'] < min(action_space)
    for a in list(action_space):
        assert df['label'] == a.value
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
        assert df['label'].max
        for stage in dataset_config.target_round:
            name = 'round_' + stage.name.casefold()
            assert df[name] == 1
        for stage in list(Stage):
            if stage not in dataset_config.target_round:
                name = 'round_' + stage.name.casefold()
                assert not df[name] == 1