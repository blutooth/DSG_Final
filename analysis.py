"""Creates features, runs classifier, generates submission."""

import pandas as pd
import pickle
import numpy as np
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from submissions import submit
from data_prepare import get_data,get_sub
from utils import Timer
from sklearn.metrics import confusion_matrix

data = pickle.load(open("./data/anal.pkl", "rb"))
data = pd.DataFrame(data.T, index=data.T[:,0], columns = ['Converted', 'Prediction'])

#
data['HardPred'] = (data.Prediction > 0.5).astype('int')

confusion_matrix(data.Converted, data.HardPred)

            preds = gbm.predict(xgb.DMatrix(X_train))
            arr = np.vstack([Y_train, preds])
            #arr = pd.DataFrame(arr)
            #arr.to_pickle('data/anal.pkl')
            pickle.dump(arr, open("data/anal.pkl", "wb"))
            #preds = gbm.predict(xgb.DMatrix(X_test))