import os
import pandas as pd
from prl.baselines.supervised_learning.training.main import run_preprocessing

dir_path = os.path.dirname(os.path.realpath(__file__))


def test_preprocessing_read_write():
    run_preprocessing(input_dir=dir_path,
                      output_dir=dir_path + "/output",
                      blind_sizes='0.25-0.50',
                      skip_preprocessing=False)
    # read csv files from output_dir
    df1 = pd.read_csv(dir_path+"/output"+"/dummy1.csv")
    df2 = pd.read_csv(dir_path+"/output"+"/dummy2.csv")
    assert df1.shape == (6, 564)
    assert df2.shape == (6, 564)


test_preprocessing_read_write()
