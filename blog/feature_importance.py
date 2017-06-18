import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tabulate import tabulate

import sys

sys.path.append(".")

import lib.features as features

if __name__ == "__main__":
    train = pd.read_csv('train.csv')

    train = features.add_features(train)

    data = train.select_dtypes(include=[np.number]).interpolate().dropna()

    y = np.log(train.SalePrice)
    X = data.drop(["SalePrice", "Id"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

    clf = RandomForestRegressor(n_jobs=2, n_estimators=1000)
    model = clf.fit(X_train, y_train)

    headers = ["name", "score"]
    values = sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: x[1] * -1)
    print(tabulate(values, headers, tablefmt="plain"))

    r_squared = model.score(X_test, y_test)
    rmse = mean_squared_error(y_test, model.predict(X_test))
    print("R^2", r_squared, "RMSE", rmse)
