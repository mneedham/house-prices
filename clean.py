import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from tabulate import tabulate

sns.set(style="whitegrid", color_codes=True)
sns.set(font_scale=1)

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


def test_model(model, X_test, y_test):
    # print("R^2 is: \n", model.score(X_test, y_test))
    # print('RMSE is: \n', mean_squared_error(y_test, model.predict(X_test)))
    return model.score(X_test, y_test), mean_squared_error(y_test, model.predict(X_test))


def enc_condition(x): return 1 if x == 'Partial' else 0


def foundation(x): return 1 if x == 'PConc' else 0


def misc_feature(x): return 1 if x == 'TenC' else 0


def fireplace(x): return 1 if x == "Ex" else 0


def exterior(x):
    if x == "Ex":
        return 5
    elif x == "Gd":
        return 4
    elif x == "TA":
        return 3
    elif x == "Fa":
        return 2
    else:
        return 1

def add_features(data):
    new_data = data.copy()
    new_data['enc_street'] = pd.get_dummies(new_data.Street, drop_first=True)
    new_data['enc_condition'] = new_data.SaleCondition.apply(enc_condition)
    new_data['enc_foundation'] = new_data.Foundation.apply(foundation)
    new_data['enc_misc_feature'] = new_data.MiscFeature.apply(misc_feature)
    new_data['enc_central_air'] = pd.get_dummies(new_data.CentralAir, drop_first=True)
    new_data['enc_fireplace'] = new_data.FireplaceQu.apply(misc_feature)
    new_data['enc_exterior'] = new_data.ExterCond.apply(exterior)
    return new_data

if __name__ == '__main__':
    headers = ["type", "R squared", "RMSE"]
    table = []

    model, X_test, y_test = create_model(train)
    table.append(["Before cleaning"] + list(test_model(model, X_test, y_test)))

    cleaned_train = clean_nulls(train)
    model, X_test, y_test = create_model(cleaned_train)
    table.append(["After cleaning"] + list(test_model(model, X_test, y_test)))

    train_extra_features = add_features(train)
    model, X_test, y_test = create_model(train_extra_features)
    table.append(["Extra features"] + list(test_model(model, X_test, y_test)))

    cleaned_train_extra_features = clean_nulls(train_extra_features)
    model, X_test, y_test = create_model(cleaned_train_extra_features)
    table.append(["Extra features after cleaning:"] + list(test_model(model, X_test, y_test)))

    print(tabulate(table, headers, tablefmt="plain"))

    # cleaned_test_extra_features = clean_nulls(add_features(test))
    #
    # submission = pd.DataFrame()
    # submission['Id'] = cleaned_test_extra_features.Id
    #
    # feats = cleaned_test_extra_features.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
    # predictions = model.predict(feats)
    # final_predictions = np.exp(predictions)
    # submission['SalePrice'] = final_predictions
    # submission.to_csv('submission-extra-features-no-nas.csv', index=False)
