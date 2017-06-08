import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid", color_codes=True)
sns.set(font_scale=1)

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')


def encode(x): return 1 if x == 'Partial' else 0


def pivot_me(index_name):
    pivot = train.pivot_table(index=index_name, values='SalePrice', aggfunc=np.median)

    sns.barplot(y="SalePrice", data=pivot)

    # pivot.plot(kind='bar', color='blue')
    # plt.xlabel(index_name)
    # plt.ylabel('Median Sale Price')
    # plt.xticks(rotation=0)
    # plt.show()


if __name__ == '__main__':
    print("Train data shape:", train.shape)
    print("Test data shape:", test.shape)

    plt.style.use(style='ggplot')
    plt.rcParams['figure.figsize'] = (10, 6)

    print("Skew is:", train.SalePrice.skew())
    # plt.hist(train.SalePrice, color='blue')
    # plt.show()

    target = np.log(train.SalePrice)
    print("Skew is:", target.skew())
    # plt.hist(target, color='blue')
    # plt.show()

    # fig, axs = plt.subplots(ncols=2)
    # sns.distplot(np.log(train["SalePrice"]), color='r', kde=False, ax=axs[0])
    # sns.distplot(train["SalePrice"], color='r', kde=False, ax=axs[1])
    # sns.plt.show()

    numeric_features = train.select_dtypes(include=[np.number])

    corr = numeric_features.corr()
    print(corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
    print(corr['SalePrice'].sort_values(ascending=False)[-5:])

    data = train.select_dtypes(include=[np.number]).interpolate().dropna()

    non_numeric_columns = set(train.columns.tolist()) - set(data.columns.tolist())
    for column in list(non_numeric_columns):
        print(column, train[column].value_counts(dropna=False).to_dict())

    # for column in list(non_numeric_columns)[:1]:
    #     pivot_me(column)
    # plt.show()

    # plotting correlations
    num_feat = train.columns[train.dtypes != object]
    num_feat = num_feat[1:-1]
    labels = []
    values = []
    for col in num_feat:
        labels.append(col)
        values.append(np.corrcoef(train[col].values, train.SalePrice.values)[0, 1])

        ind = np.arange(len(labels))

    # width = 0.9
    # fig, ax = plt.subplots(figsize=(12, 40))
    # rects = ax.barh(ind, np.array(values), color='red')
    # ax.set_yticks(ind + ((width) / 2.))
    # ax.set_yticklabels(labels, rotation='horizontal')
    # ax.set_xlabel("Correlation coefficient")
    # ax.set_title("Correlation Coefficients w.r.t Sale Price")
    # plt.show()

    correlations = train.corr()
    attrs = correlations.iloc[:-1, :-1]  # all except target

    threshold = 0.5
    important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0]) \
        .unstack().dropna().to_dict()

    unique_important_corrs = pd.DataFrame(
        list(set([(tuple(sorted(key)), important_corrs[key]) for key in important_corrs])),
        columns=['Attribute Pair', 'Correlation'])

    # sorted by absolute value
    unique_important_corrs = unique_important_corrs.ix[
        abs(unique_important_corrs['Correlation']).argsort()[::-1]]

    print(unique_important_corrs)


    # ExterCond - can be converted into an ordinal --> Po, Fa, TA, Gd, Ex
    # not quite sure how it works though - do the actual values matter as long as they're in order?
    # Foundation - one hot encoding for PConc or not
    # MiscFeature - one hot encoding for TenC or not
    # LandContour - one hot encoding for Bnk or not
    # CentralAir - convert using getDummy
    # KitchenQual - ordinal --> Po, Fa, TA, Gd, Ex
    # BsmtExposure - ordinal --> NA, No, Mn, Av, Gd
    # FireplaceQu - one hot encoding for Ex or not
    # Neighborhood - some big differences here. Maybe possible to pull in some external data to indicate area quality?
