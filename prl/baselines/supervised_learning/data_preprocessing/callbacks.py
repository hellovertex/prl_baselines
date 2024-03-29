import os
from pathlib import Path

import pandas as pd


def to_csv(df: pd.DataFrame, filename, output_dir):
    """Writes pandas.DataFrame object to .csv file in `output_dir`.
    If output_dir does not exist it will be created. `filename` must end with .csv"""
    output_dir = os.path.join(output_dir, Path(filename).parent.name)
    if not os.path.exists(output_dir):
        os.makedirs(os.path.abspath(output_dir))
    filename = os.path.basename(filename)
    df.to_csv(os.path.join(output_dir, filename), index=False)
