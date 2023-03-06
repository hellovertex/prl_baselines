from prl.baselines.supervised_learning.v2.datasets.dataset_config import ActionGenOption
from prl.baselines.supervised_learning.v2.datasets.vectorized_data import \
    make_vectorized_data_if_not_exists_already


def make_dataset_debug():
    num_top_players = 20
    nl = 'NL50'
    from_gdrive_id = '18GE6Xw4K1XE2PNiXSyh762mJ5ZCRl2SO'
    make_dataset_for_each_individual = False
    action_generation_option = ActionGenOption. \
        make_folds_from_top_players_with_randomized_hand.value
    use_multiprocessing = True,
    min_showdowns = 10,

    make_vectorized_data_if_not_exists_already(num_top_players,
                                               nl,
                                               from_gdrive_id,
                                               make_dataset_for_each_individual,
                                               action_generation_option,
                                               use_multiprocessing,
                                               min_showdowns)


if __name__ == '__main__':
    make_dataset_debug()
