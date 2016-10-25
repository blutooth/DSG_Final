import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder


def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)

def get_data():
    """Load data, mangle, add features."""

    # Load the data (500MB)
    X_train = pd.read_csv("./data/X_train.csv")
    X_test = pd.read_csv("./data/X_test.csv")

    X_train.drop(['TodayDate', 'ReceivedDateTime'], inplace=True, axis=1)
    X_test.drop(['TodayDate', 'ReceivedDateTime'], inplace=True, axis=1)

    # Add Y values to X_train. Now ignore Y_train.
    Y_train = pd.read_csv("./data/Y_train.csv")
    X_train.insert(0, "Converted", Y_train["Converted"])

    #Rename user because Pawel didn't like it
    X_train.rename(columns={'CustomerMD5Key': 'user',
                            'Unnamed: 0': 'id'}, inplace=True)
    X_test.rename(columns={'CustomerMD5Key': 'user',
                           'Unnamed: 0': 'id'}, inplace=True)


    # # Disabled as screws up test set order
    # # Sort by user (speeds up later processing).
    # if False:
    #     X_train.set_index('user')
    #     X_test.set_index('user')
    #     X_train.sort_values(by='user', inplace=True)
    #     X_test.sort_values(by='user', inplace=True)

    # Drop missing values -- only 44 rows!
    X_train.isnull().values.any()
    X_train.isnull().sum()
    X_train = X_train.dropna()

    features = set(X_train.columns) - set(['Converted', 'user'])
    features = list(features)

    X_all = X_train.copy().append(X_test)
    # Enumerate/categoricalise strings
    for f in features:
        if isinstance(X_train.loc[0, f], str) or (
                len(X_train.loc[:, f].unique()) < 0):
            print('Encoding ', f)
            lbl = LabelEncoder()
            lbl.fit(X_all.loc[:,f])
            X_train.loc[:, f] = lbl.transform(X_train.loc[:,f])
            X_test.loc[:, f] = lbl.transform(X_test.loc[:,f])


    X_train, X_test, features = add_features(X_train, X_test, features)

        #TODO improve it
    X_test.fillna(0, inplace=True)
    X_train.fillna(0, inplace=True)

    return X_train, X_test, features


def add_features(X_train, X_test, features):

    X_train, X_test, features = add_user_feature(
        X_train, X_test, features,
        'NumberQuotesFromUser', lambda x: len(x))
    # X_train, X_test, features = add_user_feature(
    #     X_train, X_test, features,
    #     'UserConvertedCount', lambda x: np.sum(x['Converted']),
    #     train_only=True)
    X_train, X_test, features = add_user_feature(
        X_train, X_test, features,
        'NoOfBrokers', lambda x: len(x['AffinityCodeId'].unique()))
    X_train, X_test, features = add_user_feature(
        X_train, X_test, features,
        'NoOfBrokersSCID', lambda x: len(x['SCID'].unique()))
    X_train, X_test, features = add_user_feature(
        X_train, X_test, features,
        'AvgVoluntaryExcess', lambda x: x['VoluntaryExcess'].mean())
    X_train, X_test, features = add_user_feature(
        X_train, X_test, features,
        'VarVoluntaryExcess', lambda x: x['VoluntaryExcess'].var())
    X_train, X_test, features = add_user_feature(
        X_train, X_test, features,
        'AvgCarInsuredValue', lambda x: x['CarInsuredValue'].mean())
    X_train, X_test, features = add_user_feature(
        X_train, X_test, features,
        'VarCarInsuredValue', lambda x: x['CarInsuredValue'].var())

    # Quote features
    features.append('DaysModYearSinceCarPurchase')
    X_train['DaysModYearSinceCarPurchase'] = X_train['DaysSinceCarPurchase'] % 365
    X_test['DaysModYearSinceCarPurchase'] = X_test['DaysSinceCarPurchase'] % 365

    features.append('YearsSinceCarPurchase')
    X_train['YearsSinceCarPurchase'] = X_train['DaysSinceCarPurchase'] / 365
    X_test['YearsSinceCarPurchase'] = X_test['DaysSinceCarPurchase'] / 365

    return X_train, X_test, features


def add_user_feature(X_train, X_test, features, column_name, user_func,
                     train_only=False):
    """Generic features adder based on a user's quotes."""
    features.append(column_name)
    try:
        X_train[column_name] = pd.read_pickle(train_pkl(column_name))
        if not train_only:
            X_test[column_name] = pd.read_pickle(test_pkl(column_name))
    except IOError:
        X_train[column_name] = np.ones(len(X_train), dtype=np.float)
        X_train = X_train.groupby('user').apply(lambda x: create_val(
            x, column_name, user_func))
        X_train[column_name].to_pickle(train_pkl(column_name))
        if not train_only:
            X_test[column_name] = np.ones(len(X_test), dtype=np.float)
            X_test = X_test.groupby('user').apply(lambda x: create_val(
                x, column_name, user_func))
            X_test[column_name].to_pickle(test_pkl(column_name))

    return X_train, X_test, features

def train_pkl(column_name):
    return '/home/jb689/dsg_finals/data/' + column_name + '.train.pkl'

def test_pkl(column_name):
    return '/home/jb689/dsg_finals/data/' + column_name + '.test.pkl'

def create_val(x, column_name, func):
    x[column_name] = func(x)
    return x


def subset_tofile():
    X_train, X_test, features = get_data()
    # get the snippet
    X_train = X_train.sort_values('user').iloc[:100000][:]
    X_test.to_csv("./data/X_test_nice.csv")
    X_test = X_test.sort_values('user').iloc[:100000][:]
    X_train.to_csv("./data/X_train_sub.csv")
    X_test.to_csv("./data/X_test_sub.csv")


def get_sub(Testsub=True):
    X_train_sub = pd.read_csv("./data/X_train_sub.csv")
    if Testsub == True:
        X_test = pd.read_csv("./data/X_test_sub.csv")
    else:
        X_test = pd.read_csv("./data/X_test_nice.csv")
    features = set(X_train_sub.columns)-set(['Converted', 'user'])
    return X_train_sub, X_test,features
