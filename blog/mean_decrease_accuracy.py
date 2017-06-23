from collections import defaultdict

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

if __name__ == "__main__":
    # boston = load_boston()
    # X = boston["data"]
    # y = boston["target"]
    # names = boston["feature_names"]
    train = pd.read_csv('train.csv')
    data = train.select_dtypes(include=[np.number]).interpolate().dropna()

    y = np.log(data.SalePrice)
    X = data.drop(["SalePrice", "Id"], axis=1)
    names = X.columns

    rf = RandomForestRegressor()
    scores = defaultdict(list)

    # crossvalidate the scores on a number of different random splits of the data
    # for train_idx, test_idx in sklearn.cross_validation.ShuffleSplit(len(X), 100, .3):
    for train_idx, test_idx in sklearn.model_selection.ShuffleSplit(test_size=.3, n_splits=100).split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        Y_train, Y_test = y.iloc[train_idx], y.iloc[test_idx]
        r = rf.fit(X_train, Y_train)
        acc = r2_score(Y_test, rf.predict(X_test))
        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t.iloc[:, i].values)
            shuff_acc = r2_score(Y_test, rf.predict(X_t))
            scores[names[i]].append((acc - shuff_acc) / acc)
    print("Features sorted by their score:")
    print(sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True))
