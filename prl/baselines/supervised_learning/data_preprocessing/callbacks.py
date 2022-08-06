import os
import pandas as pd


def to_csv(df: pd.DataFrame, filename, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(os.path.abspath(output_dir))
    filename = os.path.basename(filename)
    df.to_csv(os.path.join(output_dir, filename), index=False)
