from collections import defaultdict

import numpy as np
import sklearn.cross_validation
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

if __name__ == "__main__":
    boston = load_boston()
    X = boston["data"]
    Y = boston["target"]
    names = boston["feature_names"]

    rf = RandomForestRegressor()
    scores = defaultdict(list)


    # def __init__(self, n, n_iter=10, test_size=0.1, train_size=None,
    #              random_state=None):

    # crossvalidate the scores on a number of different random splits of the data
    # for train_idx, test_idx in sklearn.cross_validation.ShuffleSplit(len(X), 100, .3):
    for train_idx, test_idx in sklearn.model_selection.ShuffleSplit(test_size=.3, n_splits=100).split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        r = rf.fit(X_train, Y_train)
        acc = r2_score(Y_test, rf.predict(X_test))
        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = r2_score(Y_test, rf.predict(X_t))
            scores[names[i]].append((acc - shuff_acc) / acc)
    print("Features sorted by their score:")
    print(sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True))
