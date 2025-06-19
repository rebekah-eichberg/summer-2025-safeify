
from sklearn.model_selection import BaseCrossValidator
import pandas as pd
import numpy as np


def gen_splits(split_data=pd.read_parquet("../Data/CV_val_split.parquet")):
    splits=[]
    for i in range(3):
        test=split_data.reset_index(drop=True)[np.vectorize(lambda x: x==i)(split_data.cv_index)].index
        train=split_data.reset_index(drop=True)[np.vectorize(lambda x: x!=i)(split_data.cv_index)].index
        splits.append((train,test))
    return splits

class PredefinedKFold(BaseCrossValidator):
    """
    Custom cross-validator with predefined train/test splits.

    This cross-validator allows you to specify exact train/test splits,
    which is useful when you need to ensure certain groups or indices
    are always together in the same fold (e.g., for time series, group splits,
    or custom validation strategies).

    Parameters
    ----------
    split_data : pd.DataFrame
        DataFrame containing a 'cv_index' column that specifies the fold assignment
        for each row. The gen_splits function will generate splits based on this column.

    """
    def __init__(self, split_data):
        split_list=gen_splits(split_data)
        self._splits = [(np.array(train), np.array(test)) for train, test in split_list]

    def split(self, X, y=None, groups=None):
        for train_idx, test_idx in self._splits:
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self._splits)
