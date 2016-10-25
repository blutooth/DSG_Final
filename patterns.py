import pandas as pd
import pickle
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from zipfile import ZipFile

# get the data
X = pd.read_csv("/home/jb689/dsg_finals/data/X_train.csv")

X.drop(['TodayDate', 'ReceivedDateTime'], inplace=True, axis=1)
X['CustomerMD5Key'] = pd.factorize(X['CustomerMD5Key'])[0]
X.rename(columns={'CustomerMD5Key': 'user'}, inplace=True)


# new ideas:
for col in X.columns:
    print(col)
    for ii in range(10):
        print X.groupby('user').get_group(ii)[col]


    for ii in range(10):
        X.groupby('user').get_group(ii)['IsPolicyholderAHomeowner'].var()



def avgVoluntaryExcess(x):
    x['BB'] = x['VoluntaryExcess'].mean()
    #print(x.VoluntaryExcess.mean())
    return x

#X['AvgVoluntaryExcess'] = np.ones(len(X_train), dtype=np.int)
#X['AvgVoluntaryExcess'] = np.ones(len(X), dtype=np.int)
X = X.groupby('user').apply(avgVoluntaryExcess)

features.append('NumberQuotesFromUser')
X_test['NumberQuotesFromUser'] = np.ones(len(X_test), dtype=np.int)
X_test.groupby('user').apply(numberOfQuotes)