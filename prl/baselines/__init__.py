import os
# usage: from prl.baselines import DATA_DIR
DATA_DIR = os.path.abspath(
    os.path.join(
        os.path.join(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), os.pardir),
            os.pardir),
        'data')
)
assert os.path.exists(DATA_DIR)
# os.environ['environDATA_DIR'] = data_dir

