import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    train = pd.read_csv('train.csv')
    data = train.select_dtypes(include=[np.number]).interpolate().dropna()
    X = data.drop(['SalePrice', 'Id'], axis=1)
    y = np.log(train.SalePrice)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)

    lr = linear_model.LinearRegression()

    rfe = RFE(lr, 5)
    fit = rfe.fit(X_train, y_train)
    print("Features: {features}".format(features=X.columns))
    print("Num Features: {number_features}".format(number_features=fit.n_features_))
    print("Selected Features: {support}".format(support=fit.support_))
    print("Feature Ranking: {ranking}".format(ranking=fit.ranking_))

    # for column, selected in zip(X.columns, fit.support_):
    #     print(column, selected)

    selected_columns = [column for column, selected in zip(X.columns, fit.support_) if selected]
    print("Selected columns: {selected}".format(selected = selected_columns))
