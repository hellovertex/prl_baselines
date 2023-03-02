import glob
import logging
from typing import List, Type

import click
import numpy as np
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.utils import init_wrapped_env

from prl.baselines.supervised_learning.v2.datasets.dataset_options import DatasetOptions, \
    ActionGenOption
from prl.baselines.supervised_learning.v2.datasets.raw_data import TopPlayerSelector
from prl.baselines.supervised_learning.v2.fast_hsmithy_parser import \
    ParseHsmithyTextToPokerEpisode
from prl.baselines.supervised_learning.v2.datasets.tmp import EncoderV2
from prl.baselines.supervised_learning.v2.poker_model import PokerEpisodeV2


class VectorizedData:
    def __init__(self,
                 dataset_options: DatasetOptions,
                 parser_cls: Type[ParseHsmithyTextToPokerEpisode]):
        self.opt = dataset_options
        self.parser_cls = parser_cls
        self.top_player_selector = TopPlayerSelector(parser=self.parser_cls(self.opt.nl))
        self.make_player_dirs = True if self.opt.make_dataset_for_each_individual else \
            False

    def _generate_per_selected_player(self, use_multiprocessing):
        pass

    def _generate_player_pool_data(self, use_multiprocessing):
        filenames = glob.glob(f'{self.opt.dir_raw_data_top_players}**/*.txt',
                              recursive=True)
        return filenames

    def _parse_action_gen_option(self, a_opt: ActionGenOption):
        only_winners = drop_folds = fold_random_cards = None
        if a_opt.no_folds_top_player_all_showdowns:
            only_winners = False
            drop_folds = True
        elif a_opt.no_folds_top_player_all_showdowns:
            only_winners = drop_folds = True
        elif a_opt.make_folds_from_top_players_with_randomized_hand:
            fold_random_cards = True
        elif a_opt.make_folds_from_showdown_loser_ignoring_rank:
            only_winners = True
            drop_folds = False
        elif a_opt.make_folds_from_fish:
            only_winners = drop_folds = False
        return only_winners, drop_folds, fold_random_cards

    def encode_episodes(self,
                        encoder: EncoderV2,
                        episodesV2: List[PokerEpisodeV2]):
        n_episodes = len(episodesV2)
        a_opt = self.opt.action_generation_option
        only_winners, drop_folds, fold_random_cards = self._parse_action_gen_option(a_opt)
        selected_players = self.top_player_selector.get_top_n_players(self.opt.num_top_players)
        # todo: fix ordering of the selected players
        # selected_players =
        for i, ep in enumerate(episodesV2):
            print(f'Encoding episode no. {i}/{n_episodes}')
            try:
                observations, actions = encoder.encode_episode(ep,
                                                               # drop_folds=False,
                                                               drop_folds=drop_folds,
                                                               only_winners=only_winners,
                                                               limit_num_players=5,
                                                               fold_random_cards=fold_random_cards,
                                                               selected_players=selected_players,
                                                               # selected_players=['ishuha'],
                                                               verbose=False)
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

    def generate(self, dataset_options):
        parser = ParseHsmithyTextToPokerEpisode()
        env = init_wrapped_env(AugmentObservationWrapper,
                               [5000 for _ in range(6)],
                               blinds=(25, 50),
                               multiply_by=1)
        encoder = EncoderV2(env)
        filenames = glob.glob(f'{self.opt.dir_raw_data_top_players}**/*.txt',
                              recursive=True)
        for filename in filenames:
            episodesV2 = parser.parse_file(filename)
            training_data = self.encode_episodes(encoder, episodesV2)
            # todo: deprecate new_txt_to_vector_encoder and make tmp.EncoderV2 as
            #  encoder V2 the new one


@click.command()
@click.option("--num_top_players", default=20,
              type=int,
              help="How many top players hand histories should be used to generate the "
                   "data.")
@click.option("--nl",
              default='NL50',
              type=str,
              help="Which stakes the hand history belongs to."
                   "Determines the data directory.")
def main(num_top_players, nl, from_gdrive_id):

    dataset_options = DatasetOptions(num_top_players, nl)
    # raw_data = RawData(dataset_options, top_player_selector)
    # raw_data.generate(from_gdrive_id)
    vectorized_data = VectorizedData(dataset_options,
                                     parser_cls=ParseHsmithyTextToPokerEpisode)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
