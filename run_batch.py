"""Creates features, runs classifier, generates submission."""

import pandas as pd
import pickle
import numpy as np
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from submissions.submit import make_submission
from data_prepare import get_data, get_sub
from utils import Timer
from itertools import product

with Timer():
    X_train, X_test, features = get_data()
    print('Data loaded')
validate = False
RF = False
depths = [10]
etas = [0.08]  # Learning rate
alphas = [0.2]  # L1 weight penalty
num_est = [100]

log_file = open('./submissions/model_params.log', 'w+')
headers = ("eta  \t alpha  \t num_est \t depth   \t loss_train \t "
           "loss_val \t loss_test\n")
log_file.write(headers)

for depth, eta, alpha, num_est in product(depths, etas, alphas, num_est):

    xgb_params = {
        "objective": "binary:logistic",
        "num_class": 1,
        "booster": "gbtree",
        "max_depth": depth,
        "eval_metric": ["logloss", "auc"],
        "eta": eta,
        "silent": 1,
        "alpha": alpha,
    }

    loss_train = 0
    loss_val = 0
    loss_test = 0  # Placeholder to backfill test scores/compute our own.

    if validate:
        n_users = X_train['user'].unique().shape[0]
        uids = X_train['user'].unique()
        num_folds = 5
        folds = KFold(n_users, num_folds, shuffle=True)

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
                    rfc = RandomForestClassifier(n_estimators=num_est, n_jobs=-1)
                    rfc.fit(X_trainf, Y_train)

                    fidxs = np.argsort(-rfc.feature_importances_)

                    print('IMPORTANCES')
                    for i in range(len(fidxs)):
                        print(features[fidxs[i]],
                              rfc.feature_importances_[fidxs[i]])

                    preds = rfc.predict_proba(X_valf)
                    print('Log Loss ', log_loss(Y_val, preds))
                    loss_val += log_loss(Y_val, preds)

                else:
                    dtrain = xgb.DMatrix(X_trainf, np.array(Y_train))
                    dvalid = xgb.DMatrix(X_valf, np.array(Y_val))

                    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

                    gbm = xgb.train(xgb_params, dtrain, num_boost_round=num_est,
                                    evals=watchlist, early_stopping_rounds=50)

                    preds_train = gbm.predict(xgb.DMatrix(X_trainf))
                    preds_val = gbm.predict(xgb.DMatrix(X_valf))
                    loss_train += log_loss(Y_train, preds_train)
                    loss_val += log_loss(Y_val, preds_val)

        loss_train /= num_folds
        loss_val /= num_folds

    else:
        with Timer():
            Y_train = X_train.loc[:, 'Converted']
            X_train = X_train.loc[:, features]
            X_test = X_test.loc[:, features]

            if RF:
                rfc = RandomForestClassifier(n_estimators=num_est, n_jobs=-1)
                rfc.fit(X_train, Y_train)

                print('IMPORTANCES')
                fidxs = np.argsort(-rfc.feature_importances_)
                for i in range(len(fidxs)):
                    print(features[fidxs[i]], rfc.feature_importances_[fidxs[i]])
                preds = rfc.predict_proba(X_test)

            else:
                dtrain = xgb.DMatrix(X_train, np.array(Y_train))

                watchlist = [(dtrain, 'train')]

                gbm = xgb.train(xgb_params, dtrain, num_boost_round=num_est,
                                evals=watchlist, early_stopping_rounds=50)

                preds_train = gbm.predict(xgb.DMatrix(X_train))
                preds_test = gbm.predict(xgb.DMatrix(X_test))
                loss_train += log_loss(Y_train, preds_train)

                print(preds_test.shape)

            make_submission(preds_test)


    results = (
        "{:0.4f}\t {:0.4f}\t {:3d} \t{:0.6f}\t {:0.6f}\t {:0.6f}\t {:f}\n"
        .format(eta, alpha, num_est, depth, loss_train, loss_val, loss_test))
    print headers, results
    log_file.write(results)
    log_file.flush()
