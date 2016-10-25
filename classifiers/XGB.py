from matplotlib import pyplot as plt

import pandas as pd
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from matplotlib import pyplot as plt
import utils
from Data import munging

plt.interactive(True)

train_path='/home/max/PycharmProjects/DSG_final/Data/train.csv'
train = pd.read_csv(train_path, header=0)
y_train = train.pop(u'Survived')
X_train = munging.transform(train)
############################################################

print('Grid-search for parameters')
cv_params = {'max_depth': [4,6,8,10], 'min_child_weight': [3]}
ind_params = {'learning_rate': [0.1], 'n_estimators': [100], 'seed':[0], 'subsample': [.4,.6,.8,1.0], 'colsample_bytree': [.5,.75,1.0],
              'objective': 'binary:logistic'}
optimized_GBM = GridSearchCV(xgb.XGBClassifier(ind_params),
                             cv_params,
                              scoring = 'accuracy', cv = 5, n_jobs = 2)

optimized_GBM.fit(X_train, y_train)

print(optimized_GBM.grid_scores_)
############################################################
print('Grid Search CV optimized settings')
xgdmat = xgb.DMatrix(X_train, y_train)
our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
              'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':3}

cv_xgb = xgb.cv(params = our_params, dtrain = xgdmat, num_boost_round = 1000, nfold = 5,
                 metrics = ['error'], # Make sure you enter metrics inside a list or you may encounter issues!
                 early_stopping_rounds = 100) # Look for early stopping that minimizes error



our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':3}

final_gb = xgb.train(our_params, xgdmat, num_boost_round = 126)

xgb.plot_importance(final_gb)

# predicting
test_path='/home/max/PycharmProjects/DSG_final/Data/train.csv'
test = pd.read_csv(test_path, header=0)
y_test=test.pop('Survived')
X_test = munging.transform(test)
testdmat = xgb.DMatrix(X_test)
y_pred = final_gb.predict(testdmat)
y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0
y_pred

utils = reload(utils)
test_ids = test['PassengerId'].values


# test_ids = test['PassengerId'].values
# utils.write_subm(y_pred, test_ids, 'submission.csv')
