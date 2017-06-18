import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from lib import features

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

areas = {
    "Blmngtn": {"zipCode": "IA 50010"},
    "Blueste": {"zipCode": "IA 50014"},
    "BrDale": {"zipCode": "IA 50010"},
    "BrkSide": {"zipCode": "IA 50010"},
    "ClearCr": {"zipCode": "IA 50014"},
    "CollgCr": {"zipCode": "IA 50014"},
    "Crawfor": {"zipCode": "IA 50014"},
    "Edwards": {"zipCode": "IA 50014"},
    "Gilbert": {"zipCode": "IA 50105"},
    "IDOTRR": {"zipCode": "IA 50010"},
    "MeadowV": {"zipCode": "IA 50010"},
    "Mitchel": {"zipCode": "IA 50010"},
    "Names": {"zipCode": "IA 50011"},
    "NoRidge": {"zipCode": "IA 50010"},
    "NPkVill": {"zipCode": "IA 50010"},
    "NridgHt": {"zipCode": "IA 50010"},
    "NWAmes": {"zipCode": "IA 50010"},
    "OldTown": {"zipCode": "IA 50010"},
    "SWISU": {"zipCode": "IA 50011"},
    "Sawyer": {"zipCode": "IA 50014"},
    "SawyerW": {"zipCode": "IA 50014"},
    "Somerset": {"zipCode": "IA 50010"},
    "StoneBr": {"zipCode": "IA 50010"},
    "Timber": {"zipCode": "IA 50014"},
    "Veenker": {"zipCode": "IA 50011"},
}


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
    return model, X_test, y_test


def create_random_forest_model(train):
    data = train.select_dtypes(include=[np.number]).interpolate().dropna()
    y = np.log(train.SalePrice)
    X = data.drop(["SalePrice", "Id"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
    clf = RandomForestRegressor(n_jobs=2, n_estimators=1000)
    model = clf.fit(X_train, y_train)
    return model, X_test, y_test


def test_model(model, X_test, y_test):
    # print("R^2 is: \n", model.score(X_test, y_test))
    # print('RMSE is: \n', mean_squared_error(y_test, model.predict(X_test)))
    return model.score(X_test, y_test), mean_squared_error(y_test, model.predict(X_test))


def generate_predictions(test, model):
    submission = pd.DataFrame()
    submission['Id'] = test.Id
    feats = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
    predictions = model.predict(feats)
    final_predictions = np.exp(predictions)
    submission['SalePrice'] = final_predictions
    return submission

if __name__ == '__main__':
    headers = ["type", "R squared", "RMSE"]
    table = []

    cleaned_train = clean_nulls(train)
    train_extra_features = features.add_features(train)
    cleaned_train_extra_features = clean_nulls(train_extra_features)

    model, X_test, y_test = create_model(train)
    table.append(["Linear: Before cleaning"] + list(test_model(model, X_test, y_test)))

    model, X_test, y_test = create_model(cleaned_train)
    table.append(["Linear: After cleaning"] + list(test_model(model, X_test, y_test)))

    model, X_test, y_test = create_model(train_extra_features)
    table.append(["Linear: Extra features"] + list(test_model(model, X_test, y_test)))

    model, X_test, y_test = create_model(cleaned_train_extra_features)
    table.append(["Linear: Extra features after cleaning"] + list(test_model(model, X_test, y_test)))

    model, X_test, y_test = create_random_forest_model(train)
    table.append(["RF: Before cleaning"] + list(test_model(model, X_test, y_test)))

    model, X_test, y_test = create_random_forest_model(cleaned_train)
    table.append(["RF: After cleaning"] + list(test_model(model, X_test, y_test)))

    model, X_test, y_test = create_random_forest_model(train_extra_features)
    table.append(["RF: Extra features"] + list(test_model(model, X_test, y_test)))

    name = "RF: Extra features after cleaning"
    model, X_test, y_test = create_random_forest_model(cleaned_train_extra_features)
    table.append([name] + list(test_model(model, X_test, y_test)))
    friendly_file_name = name.lower().replace(" ", "-").replace(":", "")
    # 0.15151 - think this one is overfitted

    print(tabulate(table, headers, tablefmt="plain"))

    test_features = clean_nulls(features.add_features(test))
    submission = generate_predictions(test_features, model)
    submission.to_csv(friendly_file_name, index=False)
