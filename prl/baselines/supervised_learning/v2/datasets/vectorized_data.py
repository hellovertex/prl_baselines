import glob
from typing import List

import numpy as np
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env

from prl.baselines.supervised_learning.v2.datasets.dataset_options import DatasetOptions, \
    ActionGenOption
from prl.baselines.supervised_learning.v2.fast_hsmithy_parser import \
    ParseHsmithyTextToPokerEpisode
from prl.baselines.supervised_learning.v2.datasets.tmp import EncoderV2
from prl.baselines.supervised_learning.v2.poker_model import PokerEpisodeV2


class VectorizedData:
    def __init__(self, dataset_options: DatasetOptions):
        self.opt = dataset_options
        self.make_player_dirs = True if self.opt.make_dataset_for_each_individual else \
            False

    def _generate_per_selected_player(self, use_multiprocessing):
        pass

    def _generate_player_pool_data(self, use_multiprocessing):
        filenames = glob.glob(f'{self.opt.dir_raw_data_top_players}**/*.txt',
                              recursive=True)
        return filenames

    def get_selected_player_files(self):
        filenames = glob.glob(f'{self.opt.dir_raw_data_top_players}**/*.txt',
                              recursive=True)
        return filenames

    def _parse_action_gen_option(self, option):
        handle_folds = self.opt.action_generation_option.name

        assert 'no_folds' in handle_folds or 'make_folds' in handle_folds
        drop_folds = True if 'no_folds' in handle_folds else False
        only_winners = False
        if option == ActionGenOption.no_folds_top_player_only_wins:
            only_winners = True
        if option == ActionGenOption.make_folds_from_showdown_loser_ignoring_rank:
            only_winners = True
        # if option == ActionGenOption.no_folds_top_player_all_showdowns:
        #     only_winners = False
        # if option == ActionGenOption.make_folds_from_top_players_with_randomized_hand:
        #     only_winners = False
        # if option == ActionGenOption.make_folds_from_fish:
        #     only_winners = False

    def encode_episodes(self,
                        encoder: EncoderV2,
                        episodesV2: List[PokerEpisodeV2]):
        n_episodes = len(episodesV2)

        for i, ep in enumerate(episodesV2):
            print(f'Encoding episode no. {i}/{n_episodes}')
            try:
                observations, actions = encoder.encode_episode(ep,
                                                               # drop_folds=False,
                                                               drop_folds=drop_folds,
                                                               only_winners=only_winners,
                                                               limit_num_players=more_than_num_players,
                                                               randomize_fold_cards=randomize_fold_cards,
                                                               selected_players=selected_players,
                                                               # selected_players=['ishuha'],
                                                               verbose=verbose)
            except Exception as e:
                print(e)
                continue
            if not observations:
                continue
            if training_data is None:
                training_data = observations
                labels = actions
            else:
                try:
                    training_data = np.concatenate((training_data, observations),
                                                   axis=0)
                    labels = np.concatenate((labels, actions), axis=0)
                except Exception as e:
                    print(e)

    def generate(self):
        # todo: implement
        parser = ParseHsmithyTextToPokerEpisode()
        env = init_wrapped_env(AugmentObservationWrapper,
                               [5000 for _ in range(6)],
                               blinds=(25, 50),
                               multiply_by=1)
        encoder = EncoderV2(env)
        filenames = self.get_selected_player_files()
        for filename in filenames:
            episodesV2 = parser.parse_file(filename)
            # todo: fix encoder elif player.name in episode.showdown_players and not ...
            # todo: change input of encoder to be
            #  `drop_folds`
            #  `use_showdown_winner_as_target_over_selected_player`
            training_data = self.encode_episodes(encoder)
        # run on file
        # iterate each selected player
        # parse file
        # use encoder
        # per_selected_player
        # per_pool
        # 1. no_folds_top_player_only_wins
        # 2. no_folds_top_player
        # 3. make_folds_from_top_players_with_randomized_hand
        # 4. make_folds_from_showdown_loser_ignoring_rank
        # 5. make_folds_from_fish
        pass
