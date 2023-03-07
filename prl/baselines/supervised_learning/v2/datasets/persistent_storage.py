import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from prl.baselines.supervised_learning.v2.datasets.dataset_config import DatasetConfig


class PersistentStorage:
    def __init__(self,
                 dataset_options: DatasetConfig):
        self.opt = dataset_options
        self.num_files_written_to_disk = 0

    def vectorized_player_pool_data_to_disk(self,
                                            training_data: np.ndarray,
                                            labels: np.ndarray,
                                            feature_names: List[str],
                                            compression='.bz2',
                                            file_suffix='',
                                            ):
        if training_data is not None:
            columns = None
            header = False
            # write to self.opt.dir_vectorized_data
            file_path = os.path.join(self.opt.dir_vectorized_data,
                                     f'data'
                                     f'_{file_suffix}.csv{compression}')
            if not os.path.exists(Path(file_path).parent):
                os.makedirs(os.path.realpath(Path(file_path).parent), exist_ok=True)
            if not os.path.exists(file_path):
                columns = feature_names
                header = True
            df = pd.DataFrame(data=training_data,
                              index=labels,  # The index (row labels) of the DataFrame.
                              columns=columns)
            # float to int if applicable
            df = df.apply(lambda x: x.apply(lambda y: np.int8(y) if int(y) == y else y))

            df.to_csv(file_path,
                      index=True,
                      header=header,
                      index_label='label',  # index=False ?
                      mode='a',
                      float_format='%.5f',
                      compression='bz2'
                      )
            return "Success"
        return "Failure"

    def preprocessed_data_to_disk(self,
                                  df,
                                  origin_filename):
        output_dir = self.opt.dir_preprocessed_data
        if not os.path.exists(output_dir):
            os.makedirs(os.path.abspath(output_dir))
        filename = os.path.basename(origin_filename)
        filename = os.path.join(output_dir, filename)
        df.to_csv(filename, index=False)
        df.to_csv(filename + '.bz2',
                  index=True,
                  header=header,
                  index_label='label',
                  mode='a',
                  float_format='%.5f',
                  compression='bz2')
