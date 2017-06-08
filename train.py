import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')


def enc_condition(x): return 1 if x == 'Partial' else 0


def foundation(x): return 1 if x == 'PConc' else 0


def misc_feature(x): return 1 if x == 'TenC' else 0


if __name__ == '__main__':
    train = train[train['GarageArea'] < 1200]

    train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
    test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)

    train['enc_condition'] = train.SaleCondition.apply(enc_condition)
    test['enc_condition'] = test.SaleCondition.apply(enc_condition)

    train['enc_foundation'] = train.Foundation.apply(foundation)
    test['enc_foundation'] = train.Foundation.apply(foundation)

    train['enc_misc_feature'] = train.MiscFeature.apply(misc_feature)
    test['enc_misc_feature'] = train.MiscFeature.apply(misc_feature)

    train['enc_central_air'] = pd.get_dummies(train.CentralAir, drop_first=True)
    test['enc_central_air'] = pd.get_dummies(train.CentralAir, drop_first=True)

    data = train.select_dtypes(include=[np.number]).interpolate().dropna()
    y = np.log(train.SalePrice)
    X = data.drop(['SalePrice', 'Id'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

    lr = linear_model.LinearRegression()

    model = lr.fit(X_train, y_train)
    print("R^2 is: \n", model.score(X_test, y_test))

    predictions = model.predict(X_test)
    print('RMSE is: \n', mean_squared_error(y_test, predictions))

    submission = pd.DataFrame()
    submission['Id'] = test.Id

    feats = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
    predictions = model.predict(feats)
    final_predictions = np.exp(predictions)
    submission['SalePrice'] = final_predictions
    print(submission.head())
    submission.to_csv('submission1.csv', index=False)

    # [(column, train[column].value_counts(dropna=False).to_dict()) for column in set(train.columns.tolist()) - set(data.columns.tolist())]
