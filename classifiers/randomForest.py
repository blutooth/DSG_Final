from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
print(__doc__)

X = np.load('../Data/X.npy')
Y = X[:, 0]
X = X[:,1:]
Xtrain, Xtest, ytrain, ytest = train_test_split(X,
   Y, test_size=0.2, random_state=42)

rfc = RandomForestClassifier(n_estimators=20)


param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 6],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

grid_search = GridSearchCV(rfc, param_grid=param_grid)
grid_search.fit(Xtrain,ytrain)
y_pred=grid_search.predict(Xtest)
print(classification_report(ytest, y_pred))

import pickle
s = pickle.dumps(grid_search)
grid_search = pickle.loads(s)

