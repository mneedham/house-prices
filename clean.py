import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

sns.set(style="whitegrid", color_codes=True)
sns.set(font_scale=1)

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')


def find_nulls(df):
    null_columns = df.columns[df.isnull().any()]
    return df[null_columns].isnull().sum()


def clean_nulls(data):
    new_data = data.copy()

    new_data["MasVnrType"] = new_data["MasVnrType"].fillna('None')
    new_data["MasVnrArea"] = new_data["MasVnrArea"].fillna(0.0)
    new_data["Alley"] = new_data["Alley"].fillna('None')
    basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFinSF1', 'BsmtFinSF2']
    for col in basement_cols:
        if 'FinSF' not in col:
            new_data[col] = new_data[col].fillna('None')
    new_data["FireplaceQu"] = new_data["FireplaceQu"].fillna('None')
    garage_cols = ['GarageType', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea']
    for col in garage_cols:
        if new_data[col].dtype == np.object:
            new_data[col] = new_data[col].fillna('None')
        else:
            new_data[col] = new_data[col].fillna(0)
    new_data.Fence = new_data.Fence.fillna("None")
    new_data.MiscFeature = new_data.MiscFeature.fillna("None")

    missing_lot_frontage = new_data['LotFrontage'].isnull()
    new_data['SqrtLotArea'] = np.sqrt(new_data['LotArea'])
    # new_data.LotFrontage[missing_lot_frontage] = new_data.SqrtLotArea[missing_lot_frontage]

    new_data.loc[missing_lot_frontage, "LotFrontage"] = new_data.loc[missing_lot_frontage, "SqrtLotArea"]

    new_data["PoolQC"] = new_data["PoolQC"].fillna('None')
    new_data["Electrical"] = new_data["Electrical"].fillna('SBrkr')

    return new_data.drop(["SqrtLotArea"], axis=1)


def create_model(train):
    data = train.select_dtypes(include=[np.number]).interpolate().dropna()
    y = np.log(train.SalePrice)
    X = data.drop(['SalePrice', 'Id'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
    lr = linear_model.LinearRegression()
    model = lr.fit(X_train, y_train)
    return (model, X_test, y_test)


def test_model(model, X_test, y_test):
    print("R^2 is: \n", model.score(X_test, y_test))
    print('RMSE is: \n', mean_squared_error(y_test, model.predict(X_test)))


if __name__ == '__main__':
    print("Before cleaning:")
    model, X_test, y_test = create_model(train)
    test_model(model, X_test, y_test)

    print("After cleaning:")
    cleaned_train = clean_nulls(train)
    model, X_test, y_test = create_model(cleaned_train)
    test_model(model, X_test, y_test)

    submission = pd.DataFrame()
    submission['Id'] = test.Id

    feats = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
    predictions = model.predict(feats)
    final_predictions = np.exp(predictions)
    submission['SalePrice'] = final_predictions
    submission.to_csv('submission-no-nas.csv', index=False)
