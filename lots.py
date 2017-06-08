import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

train = pd.read_csv('train.csv')

if __name__ == "__main__":
    null_columns = train.columns[train.isnull().any()]
    print(train[null_columns].isnull().sum())

    cols = [col for col in train.columns if col.startswith("Lot")]
    missing_frontage = train[cols][train["LotFrontage"].isnull()]
    print(missing_frontage.head())

    # what lot configurations do we have?
    # print(train.groupby(["LotConfig"])["Id"].count())

    # can we predict what the missing LotFrontage values should be from the other values we do have?

    # train["LotFrontageTransformed"] = train["LotFrontage"].replace(np.nan, -1).apply(lambda x: x > -1)
    # print(train.groupby(["LotFrontageTransformed"])["Id"].count())

    # what's the distribution of the LotFrontage values?

    # split the training set based on whether the LotFrontage value is NaN
    sub_train = train[train.LotFrontage.notnull()]

    # dummies = pd.get_dummies(sub_train[cols].LotShape)
    # print(dummies.head())

    data = pd.concat([
        sub_train[cols],
        pd.get_dummies(sub_train[cols].LotShape),
        pd.get_dummies(sub_train[cols].LotConfig)
    ], axis=1).select_dtypes(include=[np.number])

    print(data.head())

    X = data.drop(["LotFrontage"], axis=1)
    y = data.LotFrontage

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

    lr = linear_model.LinearRegression()

    model = lr.fit(X_train, y_train)

    print("R^2 is: \n", model.score(X_test, y_test))

    predictions = model.predict(X_test)
    print('RMSE is: \n', mean_squared_error(y_test, predictions))

    sub_test = train[train.LotFrontage.isnull()]
