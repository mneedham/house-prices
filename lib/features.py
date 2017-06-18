import pandas as pd


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
