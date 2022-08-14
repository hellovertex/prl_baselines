import os

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

    def write_train_data(self, data, labels, feature_names, n_samples, blind_sizes):
        file_dir = os.path.join(str(DATA_DIR) + "/02_vectorized/", blind_sizes)
        self.num_lines_written += n_samples
        # create new file every 500k lines
        file_name = self.out_filename_base + '_' + str(int(self.num_lines_written / 500000)) + '.csv'
        file_path = os.path.join(file_dir, file_name)
        columns = None
        index = False
        header = False
        if not os.path.exists(file_path):
            os.makedirs(os.path.realpath(file_dir), exist_ok=True)
            columns = feature_names
            index = True
            header = True
        pd.DataFrame(data=data,
                     index=labels,
                     columns=columns).to_csv(
            file_path, index=index, header=header, index_label='label', mode='a')
        return file_dir, file_path
