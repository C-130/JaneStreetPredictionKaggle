"""The code for loading, training, and cross-validation."""
from typing import Optional, Any, Callable

import numpy as np
import xgboost as xgb
import pandas as pd
import polars as pl
import hyperopt
from hyperopt import hp, Trials, tpe, fmin
import matplotlib.pyplot as plt

DATA_DIR = "jane-street-real-time-market-data-forecasting"

XGB_PARAMS = {'n_estimators': 500,
              'learning_rate': 0.1,
              'max_depth': 8,
              'subsample': 0.8,
              'lambda': 1.0,
              'alpha': 0,
              'random_state': 8,
              'tree_method': 'auto',
              'early_stopping_rounds': 20}

PARAM_TYPES = {'n_estimators': 'int',
               'learning_rate': 'float',
               'max_depth': 'int',
               'subsample': 'float',
               'lambda': 'float',
               'alpha': 'float',
               'random_state': 'int',
               'tree_method': 'str',
               'early_stopping_rounds': 'int'}

SEARCH_SPACE = {'n_estimators': hp.quniform('n_estimators', 300, 1200, 200),
                'learning_rate': hp.loguniform('learning_rate', np.log(0.02), np.log(0.25)),
                'max_depth': hp.quniform('max_depth', 5, 11, 1),
                'subsample': hp.quniform('subsample', 0.6, 0.95, 0.1),
                'lambda': hp.loguniform('lambda', np.log(0.1), np.log(1.5)),
                'alpha': hp.quniform('alpha', 0.0, 1.0, 0.2),
                'random_state': 8,
                'tree_method': 'auto',
                'early_stopping_rounds': 20}

FEATURE_COLUMNS_XGB = ['date_id', 'time_id', 'symbol_id']
FEATURE_COLUMNS_XGB.extend([f'feature_{i:02d}' for i in range(79)])


def load_data(data_dir: str,
              start_partition_id: int = 0,
              end_partition_id: int = 9,
              df_type: str = 'pl',
              fillna_opt: Optional[str | dict[str, str]] = None) -> pl.DataFrame | pd.DataFrame:
    """Load training data."""
    if start_partition_id == 0 and end_partition_id == 9:
        paths = data_dir + '/train.parquet'
    else:
        paths = [data_dir + f'/train.parquet/partition_id={i}/part-0.parquet'
                 for i in range(start_partition_id, end_partition_id + 1)]
    print('Loading data...')
    if df_type == 'pl' and fillna_opt:
        return pl.scan_parquet(paths).collect().fill_null(strategy=fillna_opt)
    elif df_type == 'pl':
        return pl.scan_parquet(paths).collect()
    elif df_type == 'pd' and fillna_opt:
        return pl.scan_parquet(paths).collect().fill_null(strategy=fillna_opt).to_pandas()
    elif df_type == 'pd':
        return pl.scan_parquet(paths).collect().to_pandas()
    else:
        raise ValueError(f'Invalid arguments')


def kfold_validation_split(data: pl.DataFrame,
                           num_fold: int = 9,
                           by: str = 'index',
                           gap: int = 0,
                           df_type: str = 'pl') \
        -> list[tuple[pd.DataFrame | pl.DataFrame, pd.DataFrame | pl.DataFrame]]:
    """Generate purged group train-validation split."""
    assert len(data) > num_fold + 1
    res = []
    if by == 'index':
        fold_size = len(data) // (num_fold + 1)
        assert gap < fold_size
        endpoints = np.arange(fold_size, len(data), fold_size)
        endpoints[-1] = len(data) - 1
        assert len(endpoints) == num_fold + 1
        for j in range(num_fold):
            print(f"Producing {j + 1}th fold...")
            data_train = data[:endpoints[j]]
            data_val = data[endpoints[j] + gap: endpoints[j + 1]]
            res.append((data_train, data_val))
    elif by == 'date':
        date_ids = data['date_id'].unique()
        fold_size = len(date_ids) // (num_fold + 1)
        assert gap < fold_size
        endpoints = np.arange(fold_size, len(date_ids), fold_size)
        endpoints[-1] = date_ids[-1]
        assert len(endpoints) == num_fold + 1
        for j in range(num_fold):
            data_train = data.filter(pl.col('date_id') < endpoints[j])
            data_val = data.filter(endpoints[j] + gap <= pl.col('data_id') < endpoints[j + 1])
            res.append((data_train, data_val))
    if df_type == 'pl':
        return res
    elif df_type == 'pd':
        return [(dtrain.to_pandas(), dval.to_pandas()) for dtrain, dval in res]
    else:
        raise ValueError(f'Unsupported data type: {df_type}')


def get_kfold_data_for_xgb(data: pl.DataFrame, k: int, gap: int) -> \
        tuple[list[pd.DataFrame], list[pd.Series], list[pd.DataFrame], list[pd.Series], list[pd.Series]]:
    """Get the list of kfold training and validation data that an XGBoost model needs."""
    kfold = kfold_validation_split(data, num_fold=k, gap=gap)
    X_train_data = []
    y_train_data = []
    X_val_data = []
    y_val_data = []
    weight_val_data = []
    for data_train, data_val in kfold:
        X_train_data.append(data_train.select(FEATURE_COLUMNS_XGB).to_pandas())
        y_train_data.append(data_train['responder_6'].to_pandas())
        X_val_data.append(data_val.select(FEATURE_COLUMNS_XGB).to_pandas())
        y_val_data.append(data_val['responder_6'].to_pandas())
        weight_val_data.append(data_val['weight'].to_pandas())
    return X_train_data, y_train_data, X_val_data, y_val_data, weight_val_data


def kfold_generator(data: pd.DataFrame,
                    feature_columns: list[str],
                    num_fold: int,
                    gap: int,
                    by: str = 'index'):
    """A generator that yields the kfold training and validation data."""
    if by == 'index':
        fold_size = len(data) // num_fold
        endpoints = np.arange(fold_size, len(data), fold_size)
        endpoints[-1] = len(data) - 1
        for j in range(num_fold):
            data_train = data[:endpoints[j]]
            data_val = data[endpoints[j] + gap: endpoints[j + 1]]
            yield data_train[feature_columns], data_train['responder_6'], \
                data_val[feature_columns], data_val['responder_6'], data_val['weight']
    elif by == 'date':
        date_ids = data['date_id'].unique()
        fold_size = len(date_ids) // (num_fold + 1)
        endpoints = np.arange(fold_size, len(date_ids), fold_size)
        endpoints[-1] = date_ids[-1]
        for j in range(num_fold):
            data_train = data.loc[data['date_id'] < date_ids[j]]
            data_val = data.loc[(date_ids[j] + gap <= data['date_id']) & (data['date_id'] < date_ids[j + 1])]
            yield data_train[feature_columns], data_train['responder_6'], \
                data_val[feature_columns], data_val['responder_6'], data_val['weight']
    else:
        raise ValueError("Invalid argument.")


def weighted_uncentered_r2(y_val: np.ndarray | pd.Series,
                           y_pred: np.ndarray | pd.Series,
                           weights: np.ndarray | pd.Series) -> float:
    """Compute the weighted uncentered r^2 score."""
    return 1 - sum(weights * (y_val - y_pred) ** 2) / sum(weights * y_val ** 2)


def train_and_validate(data: pd.DataFrame,
                       k: int,
                       gap: int,
                       hyperparams: dict[str, Any],
                       verbose=True) -> tuple[xgb.XGBRegressor, float]:
    """Train and validate a model based on the data and hyperparameters."""
    model = xgb.XGBRegressor(**hyperparams)
    scores = []
    total_vals = 0
    val_counts = []
    cv_gen = kfold_generator(data, FEATURE_COLUMNS_XGB, k, gap)
    for i in range(k):
        if verbose:
            print(f"Starts training on {i + 1}th fold")
        X_train, y_train, X_val, y_val, weights = next(cv_gen)
        model.fit(X_train, y_train)
        scores.append(weighted_uncentered_r2(y_val, model.predict(X_val), weights))
        total_vals += len(y_val)
        val_counts.append(len(y_val))
        if verbose:
            print(f"Finished training on {i + 1}th fold")
    score_weights = [score / total_vals for score in scores]
    return model, sum(score_weights[j] * scores[j] for j in range(k))


def train_and_validate_models(data: pd.DataFrame,
                              k: int,
                              gap: int,
                              hyperparam_set: list[dict[str, Any]]) -> list[tuple[xgb.XGBRegressor, float]]:
    """Create a group of models from the list of hyperparameter configs and evaluate them on the validation set."""
    res = []
    num = 0
    for hyperparams in hyperparam_set:
        print(f"Start training {num + 1}th model.")
        res.append(train_and_validate(data, k, gap, hyperparams, False))
        print(f"Finished training {num + 1}th model.")
    return res


def objective(space: dict[str, Any], data: pd.DataFrame, k: int, gap: int) -> float:
    """The objective function for hyperopt."""
    hyperparams = {}
    for param in space:
        if PARAM_TYPES[param] == 'int':
            hyperparams[param] = int(space[param])
        else:
            hyperparams[param] = space[param]
    _, eval_score = train_and_validate(data, k, gap, hyperparams, False)
    print(f"r^2 score: {eval_score}")
    return eval_score


def hyperopt_cross_validation(data: pd.DataFrame,
                              space: dict[str, Any],
                              k: int,
                              gap: int,
                              objective: Callable) -> tuple[dict[str, Any], Trials]:
    """Cross validation using hyperopt."""
    trials = Trials()
    optimal = fmin(
        fn=lambda hpm: objective(hpm, data, k, gap),
        space=space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials)
    return optimal, trials


def finalize_model(data: pd.DataFrame,
                   hyperparams: dict[str, Any],
                   write: bool = False,
                   write_path: str = '') -> xgb.XGBRegressor:
    """Finalize the model based on the optimal hyperparameters."""
    X_train = data[FEATURE_COLUMNS_XGB]
    y_train = data['responder_6']
    model = xgb.XGBRegressor(**hyperparams)
    model.fit(X_train, y_train)
    if write:
        model.save_model(write_path)
    return model


if __name__ == '__main__':
    df = load_data(DATA_DIR, fillna_opt={'feature_01': 'mean', 'feature_03': 'zero'})
    # date_counts = df.group_by('date_id').len()
    # plt.scatter(date_counts['date_id'].to_numpy(), date_counts['len'].to_numpy())
    # plt.xlabel('date_id')
    # plt.ylabel('number of data')
    # plt.title('distribution of data over date_id')
    # plt.show()
