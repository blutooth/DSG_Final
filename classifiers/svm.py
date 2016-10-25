from __future__ import print_function

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np
print(__doc__)

# Loading the Digits dataset
X = np.load('../Data/X.npy')
Y = X[:, 0]
X = X[:,1:]
Xtrain, Xtest, ytrain, ytest = train_test_split(X,
   Y, test_size=0.2, random_state=42)


# Set the parameters by cross-validation
tuned_parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}
scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1,cache_size=50), tuned_parameters, cv=5,
                       scoring='%s_weighted' % score)
    clf.fit(Xtrain, ytrain)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = ytest, clf.predict(Xtest)
    print(classification_report(y_true, y_pred))
    print()
