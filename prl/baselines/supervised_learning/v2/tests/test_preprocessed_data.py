import glob
import os
from pathlib import Path

import pandas as pd
import pytest

from prl.baselines.supervised_learning.v2.datasets.dataset_config import ActionGenOption, \
    DatasetConfig
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


def test_preprocessor(dataset_config):
    make_preprocessed_data_if_not_exists_already(dataset_config,
                                                 use_multiprocessing=True)
    csv_files = glob.glob(dataset_config.dir_vectorized_data + '/**/*.csv.bz2',
                          recursive=True)
    for file in csv_files:
        df = pd.read_csv(file,
                         sep=',',
                         dtype='float32',
                         encoding='cp1252',
                         compression='bz2')
        assert df['label'].max
