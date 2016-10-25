"""
Draft for stacking

"""
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


def run(filepath):
    X = np.load(filepath)
    y = X[:, 0]
    X = X[:, 1::]
    Y = LabelEncoder().fit_transform(y)


    # The DEV SET will be used for all training and validation purposes
    # The TEST SET will never be used for training, it is the unseen set.
    X_dev, X_test, Y_dev, Y_test = train_test_split(X, y, test_size=0.10, random_state=36)

    n_trees = 250
    n_folds = 5

    # Our level 0 classifiers
    clfs = [
        RandomForestClassifier(n_estimators=n_trees, criterion='gini'),
        ExtraTreesClassifier(n_estimators=n_trees * 2, criterion='gini'),
        GradientBoostingClassifier(n_estimators=n_trees),


    ]

    # Ready for cross validation
    skf = list(StratifiedKFold(Y_dev, 5))

    # Pre-allocate the data
    blend_train = np.zeros((X_dev.shape[0], len(clfs)))  # Number of training data x Number of classifiers
    blend_test = np.zeros((X_test.shape[0], len(clfs)))  # Number of testing data x Number of classifiers

    print 'X_test.shape = %s' % (str(X_test.shape))
    print 'blend_train.shape = %s' % (str(blend_train.shape))
    print 'blend_test.shape = %s' % (str(blend_test.shape))
    scores = []
    # For each classifier, we train the number of fold times (=len(skf))
    for j, clf in enumerate(clfs):
        print 'Training classifier [%s]' % (j)
        blend_test_j = np.zeros((X_test.shape[0], len(
            skf)))  # Number of testing data x Number of folds , we will take the mean of the predictions later
        for i, (train_index, cv_index) in enumerate(skf):
            print 'Fold [%s]' % (i)

            # This is the training and validation set
            X_train = X_dev[train_index]
            Y_train = Y_dev[train_index]
            X_cv = X_dev[cv_index]
            Y_cv = Y_dev[cv_index]

            clf.fit(X_train, Y_train)

            # This output will be the basis for our blended classifier to train against,
            # which is also the output of our classifiers
            blend_train[cv_index, j] = clf.predict(X_cv)
            blend_test_j[:, i] = clf.predict(X_test)
        # Take the mean of the predictions of the cross validation set
        blend_test[:, j] = blend_test_j.mean(1)
        scores.append(metrics.accuracy_score(Y_test, (blend_test_j.mean(1) > 0.5).astype('int')))
    print 'Y_dev.shape = %s' % (Y_dev.shape)

    # Start blending!
    bclf = LogisticRegression()
    bclf.fit(blend_train, Y_dev)

    # Predict now
    Y_test_predict = bclf.predict(blend_test)
    score = metrics.accuracy_score(Y_test, Y_test_predict)
    print 'Accuracy = %s others = %s' % (score, scores)

    return score


if __name__ == '__main__':
    run('X.npy')
