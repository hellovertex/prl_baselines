import os
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from tensorboardX import SummaryWriter


class TensorBoardCallback(xgb.callback.TrainingCallback):
    def __init__(self, experiment: str = None, data_name: str = None):
        self.experiment = experiment or "logs"
        self.data_name = data_name or "test"
        self.datetime_ = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = f"runs/{self.experiment}/{self.datetime_}"
        self.train_writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "train/"))
        if self.data_name:
            self.test_writer = SummaryWriter(
                log_dir=os.path.join(self.log_dir, f"{self.data_name}/")
            )

    def after_iteration(
            self, model, epoch: int, evals_log: xgb.callback.TrainingCallback.EvalsLog
    ) -> bool:
        if not evals_log:
            return False

        for data, metric in evals_log.items():
            for metric_name, log in metric.items():
                score = log[-1][0] if isinstance(log[-1], tuple) else log[-1]
                if data == "train":
                    self.train_writer.add_scalar(metric_name, score, epoch)
                else:
                    self.test_writer.add_scalar(metric_name, score, epoch)

        return False


from sklearn.metrics import classification_report


def precision(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    a = 1
    return 1

def recall(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    a = 1
    return 1

if __name__ == '__main__':
    filename = "/home/hellovertex/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/training/scratches/top_100_only_wins_no_folds_per_round/Round_preflop/data.csv.bz2"
    tmp = pd.read_csv(filename,
                     # df = pd.read_csv(path_to_csv_files,
                     sep=',',
                     dtype='float32',
                     #chunksize=1000,
                     # dtype='float16',
                     encoding='cp1252',
                     compression='bz2')
    #tmp = next(df).apply(pd.to_numeric, downcast='integer', errors='coerce').dropna()
    tmp = tmp.apply(pd.to_numeric, downcast='integer', errors='coerce').dropna()
    tmp = tmp.sample(frac=1)
    # remove duplicate label column
    tmp.pop('label')

    # one hot encode button
    one_hot_btn = pd.get_dummies(tmp['btn_idx'], prefix='btn_idx')
    tmp = pd.concat([tmp, one_hot_btn], axis=1)
    tmp.drop('btn_idx', axis=1, inplace=True)
    tmp.to_csv('/home/hellovertex/Documents/github.com/prl_baselines/prl/baselines/supervised_learning/v2/preflop_all_players/data.csv.bz2',
               compression='bz2')
    # make train test split
    xtest = tmp.sample(frac=0.2, axis=0)
    xtrain = tmp.drop(index=xtest.index)

    ytest = xtest.pop('label.1')
    ytrain = xtrain.pop('label.1')

    # make Dmats
    # mat_train = xgb.DMatrix(xg_train, label=ytrain)
    xg_train = xgb.DMatrix(xtrain, label=ytrain)
    # mat_test = xgb.DMatrix(xg_test, label=ytest)
    xg_test = xgb.DMatrix(xtest, label=ytest)

    # setup parameters for xgboost
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softprob'  # 'multi:softmax'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['nthread'] = 4
    param['num_class'] = 8
    param['eval_metric'] = ['auc', 'ams@0', 'mlogloss', 'merror']
    param['custom_metric'] = [precision, recall]
    # params to consider to make the model more conservative
    # -- lambda: L2 reg weights, default=1 ;; alpha: L1 Reg weights, default = 0

    watchlist = [(xg_train, 'train'), (xg_test, 'test')]
    num_round = 5000
    callbacks = [TensorBoardCallback()]
    eval_results = {}
    bst = xgb.train(params=param,
                    dtrain=xg_train,
                    num_boost_round=num_round,
                    evals=watchlist,
                    evals_result=eval_results,  # stores evaluation of metrics for `evals`
                    # custom_metric=[precision, recall],  # set maximize=True to maximize custom metric
                    callbacks=[TensorBoardCallback(experiment='exp_1', data_name='test')])
    a = 1
