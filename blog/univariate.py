import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from tabulate import tabulate

if __name__ == "__main__":
    train = pd.read_csv('train.csv')
    data = train.select_dtypes(include=[np.number]).interpolate().dropna()
    X = data.drop(['SalePrice', 'Id'], axis=1)
    y = np.log(train.SalePrice)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

    test = SelectKBest(score_func=f_regression, k=4)
    fit = test.fit(X, y) # ValueError: Unknown label type - https://stackoverflow.com/questions/34246336/python-randomforest-unknown-label-error - only when using chi2 function
    # fit = test.fit(X, np.asarray(y, dtype="|S6"))

    np.set_printoptions(precision=3, suppress=True)

    print("Features: {features}".format(features=X.columns))
    print("Scores: {scores}".format(scores=fit.scores_))

    values = [(value, float(score)) for value, score in sorted(zip(X.columns, fit.scores_), key=lambda x: x[1] * -1)]
    print(tabulate(values, ["column", "score"], tablefmt="plain", floatfmt=".4f"))


    selected_features = fit.transform(X)
    print("Features: {selected_features}".format(selected_features = selected_features))
