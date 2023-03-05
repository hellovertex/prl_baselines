import os

import numpy as np
import pandas as pd

from core.writer import Writer
# from azureml.core import Experiment
# from azureml.core import Workspace
from prl.baselines.supervised_learning.config import DATA_DIR


class CSVWriter(Writer):
    """This handles creation and population of training data folders, containing encoded
    PokerEpisode instances. These encodings can be used for supervised learning. """

    def __init__(self, out_filename_base):
        super(CSVWriter).__init__()
        self.n_files_written_this_run = 0
        self.out_filename_base = out_filename_base
        self.num_lines_written = 0

    def write_train_data(self, data, labels, feature_names, n_samples, subdir):
        file_dir = os.path.join(str(DATA_DIR) + "/02_vectorized/", subdir)
        self.num_lines_written += n_samples
        # create new file every 100k lines
        file_name = self.out_filename_base + '_' + str(
            int(self.num_lines_written / 100000)) + '.csv'
        file_path = os.path.join(file_dir, file_name)
        columns = None
        header = False
        if not os.path.exists(file_path):
            os.makedirs(os.path.realpath(file_dir), exist_ok=True)
            columns = feature_names
            header = True
        df = pd.DataFrame(data=data,
                          index=labels,  # The index (row labels) of the DataFrame.
                          columns=columns)
        # float to int if applicable
        df = df.apply(lambda x: x.apply(lambda y: np.int8(y) if int(y) == y else y))
        df.to_csv(file_path + '.bz2',
                  index=True,
                  header=header,
                  index_label='label',
                  mode='a',
                  float_format='%.5f',
                  compression='bz2')
        return file_dir, file_path
