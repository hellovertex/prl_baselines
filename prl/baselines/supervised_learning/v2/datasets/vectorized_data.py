from prl.baselines.supervised_learning.v2.datasets.dataset_options import DatasetOptions


class VectorizedData:
    def __init__(self, dataset_options: DatasetOptions):
        self.opt = dataset_options
        self.make_player_dirs = True if self.opt.make_dataset_for_each_individual else \
            False
        
    def _generate_per_selected_player(self):
        pass

    def _generate_player_pool_data(self):
        pass

    def generate(self):
        # todo: implement
        # use encoder
        # per_selected_player
        # per_pool
        # 1. no_folds_top_player_wins
        # 2. no_folds_top_player
        # 3. make_folds_from_top_players_with_randomized_hand
        # 4. make_folds_from_showdown_loser
        # 5. make_folds_from_non_top_player
        pass
