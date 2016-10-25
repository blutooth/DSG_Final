"""Creates features, runs classifier, generates submission."""

import pandas as pd
import pickle
import numpy as np
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from submissions import submit
from data_prepare import get_data, get_sub
from utils import Timer
from sklearn.preprocessing import OneHotEncoder


def postProProbs(df):
    def desmooth(x):
        probs = np.array(x['Preds'])
        sidx = np.argsort(probs)
        l = len(sidx)
        post_probs = np.zeros((l,))
        for i in range(l):
            if i < l*0.6 and l > 10:
                post_probs[sidx[i]] = 0.8*probs[sidx[i]]
                #print('PS', i, l, post_probs[sidx[i]])
            else:
                post_probs[sidx[i]] = probs[sidx[i]]
                #print('PL', i, l, post_probs[sidx[i]])
        post_probs[post_probs>1] = 1.0
        x['PostPreds'] = post_probs
        return x

    df.insert(0, "PostPreds", np.ones(len(df)))
    df = df.groupby('user').apply(desmooth)
    #print(df.columns)
    return df['PostPreds']

X_train, X_test, features = get_data()
print('Data loaded')
validate = True
RF = False

if validate:
    n_users = X_train['user'].unique().shape[0]

    uids = X_train['user'].unique()
    folds = KFold(n_users, 3, shuffle=True)

    buys = X_train.groupby('user')['Converted'].sum()
    is_buyer = np.array(buys) > 0
    uids = np.array(buys.index)

    folds = StratifiedKFold(is_buyer, 4, shuffle=True)
    for train_idx, test_idx in folds:
        with Timer():
            train_uids = uids[train_idx]
            test_uids = uids[test_idx]
            assert(len(set(train_uids).intersection(set(test_uids))) == 0)

            X_trainv = X_train.loc[X_train['user'].isin(train_uids), :]
            X_val = X_train.loc[X_train['user'].isin(test_uids), :]

            Y_train = X_trainv['Converted']
            Y_val = X_val['Converted']

            X_trainf = X_trainv.loc[:, features]
            X_trainf = X_trainv.loc[:, features]
            X_valf = X_val.loc[:, features]
            if RF:
                rfc = RandomForestClassifier(n_estimators=10, n_jobs=-1)
                rfc.fit(X_trainf, Y_train)

                fidxs = np.argsort(-rfc.feature_importances_)

                print('IMPORTANCES')
                for i in range(len(fidxs)):
                    print(features[fidxs[i]], rfc.feature_importances_[fidxs[i]])

                preds = rfc.predict_proba(X_valf)
                print('Log Loss ', log_loss(Y_val, preds))

            else:
                w = np.ones((len(Y_train),1))
                w[np.array(Y_train) == 1] = 2.
                dtrain = xgb.DMatrix(X_trainf, np.array(Y_train), weight=w)
                dvalid = xgb.DMatrix(X_valf, np.array(Y_val))


                params = {
                    "objective": "binary:logistic",
                    "num_class": 1,
                    "booster": "gbtree",
                    "max_depth": 10,  # 5 - 15
                    "eval_metric": "logloss",
                    "eta": 0.08, # lr
                    "silent": 1,
                    "alpha": 0.2,
                    "gamma":4,
                    "n_estimators": 300,  # 100 - 1000
                    "max_delta_step":1,
                    "colsample_bytree":0.9
                }
                #eval_metric = ["logloss", "auc", "map"],
                watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
                gbm = xgb.train(params, dtrain, 500,  evals=watchlist,
                                early_stopping_rounds=100)

else:
    with Timer():
        Y_train = X_train.loc[:, 'Converted']
        X_train = X_train.loc[:, features]
        X_test_f = X_test
        X_test = X_test.loc[:, features]

        if RF:
            rfc = RandomForestClassifier(n_estimators=600, n_jobs=-1)
            rfc.fit(X_train, Y_train)

            print('IMPORTANCES')
            fidxs = np.argsort(-rfc.feature_importances_)
            for i in range(len(fidxs)):
                print(features[fidxs[i]], rfc.feature_importances_[fidxs[i]])
            preds = rfc.predict_proba(X_test)
            submit.make_submission(preds[:,1])


        else:
            try:
                preds =  np.load('preds.npy')
            except:
                w = np.ones((len(Y_train),1))

                w[np.array(Y_train) == 1] = 4.
                dtrain = xgb.DMatrix(X_train, np.array(Y_train), weight=w)

                params = {
                    "objective": "binary:logistic",
                    "num_class": 1,
                    "booster": "gbtree",
                    "max_depth": 10,  # 5 - 15
                    "eval_metric": "logloss",
                    "eta": 0.08, # lr
                    "silent": 1,
                    "alpha": 0.2,
                    "gamma":4,
                    "n_estimators": 300,  # 100 - 1000
                    "max_delta_step":1,
                    "colsample_bytree":0.9,
                    "nthread":40
                }

                watchlist = [(dtrain, 'train')]
                gbm = xgb.train(params, dtrain, 300, evals=watchlist,
                                early_stopping_rounds=100)
                preds = gbm.predict(xgb.DMatrix(X_test))
                np.save('preds.npy', preds)

            X_test_f.insert(0, "Preds", preds)
            #preds = postProProbs(X_test_f)
            submit.make_submission(preds)
