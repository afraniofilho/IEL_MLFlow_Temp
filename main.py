# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Ler arquivo
    csv_url = ("C:\ML\jena_climate_2009_2016.csv"
    )
    df = pd.read_csv(csv_url, index_col="Date Time", parse_dates=True)

    df = df.drop(index=df.index[0:5])
    df.drop_duplicates(inplace=True)
    df = df.asfreq('H')
    df['T (degC) 2 horas depois'] = df['T (degC)'].shift(-2)
    df.fillna(method="ffill", inplace=True)

    train = df.loc[df.index < '01-01-2015']
    test = df.loc[df.index >= '01-01-2015']

    FEATURES = ['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
                'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
                'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
                'wd (deg)']
    TARGET = 'T (degC) 2 horas depois'

    train_x = train[FEATURES]
    train_y = train[TARGET]

    test_x = test[FEATURES]
    test_y = test[TARGET]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5

    with mlflow.start_run():
        reg = Ridge(alpha=alpha)

        reg.fit(train_x, train_y)

        predic = reg.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predic)

        print("Ridge model (alpha={:f}):".format(alpha))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(reg, "model", registered_model_name="Temp_Predic")
        else:
            mlflow.sklearn.log_model(reg, "model")