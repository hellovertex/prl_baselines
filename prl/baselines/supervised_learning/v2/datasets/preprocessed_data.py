import logging

from prl.baselines.supervised_learning.v2.datasets.dataset_options import DatasetOptions


class PreprocessedData:
    pass


def main():
    # parser = ParseHsmithyTextToPokerEpisode(nl=nl)
    dataset_options = DatasetOptions()
    # top_player_selector = TopPlayerSelector(parser)
    # raw_data = RawData(dataset_options, top_player_selector)
    # raw_data.generate(from_gdrive_id)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
