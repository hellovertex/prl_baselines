import glob
import logging
import os

import numpy as np
import pandas as pd

from prl.baselines.supervised_learning.v2.datasets.dataset_options import DatasetOptions


class PreprocessedData:

    def __init__(self,
                 dataset_options: DatasetOptions):
        self.opt = dataset_options

    def generate(self):
        if os.path.exists(self.opt.dir_preprocessed_data):
            logging.info(f'Preprocessed data already exists at directory '
                         f'{self.opt.dir_preprocessed_data} '
                         f'for given configuration: {self.opt}')
        else:
            # load .csv files into dataframe
            csv_files = glob.glob(self.opt.dir_vectorized_data + '**/*.csv',
                                  recursive=True)
            for file in csv_files:
                df = pd.read_csv(file, sep=',',
                                 dtype='float32',
                                 encoding='cp1252')
                # float to int if applicable
                df = df.apply(
                    lambda x: x.apply(lambda y: np.int8(y) if int(y) == y else y))
                # int64 to int8 to save memory
                df = df.apply(pd.to_numeric, downcast='integer', errors='coerce').dropna()

                # maybe remove unused round
                # maybe reduce action space
                # write to disk

def main():
    # parser = ParseHsmithyTextToPokerEpisode(nl=nl)
    dataset_options = DatasetOptions()
    # top_player_selector = TopPlayerSelector(parser)
    # raw_data = RawData(dataset_options, top_player_selector)
    # raw_data.generate(from_gdrive_id)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
