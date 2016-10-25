import pandas as pd
import pickle
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from zipfile import ZipFile

# get the data
X_train = pd.read_csv("/home/mec68/DSG_data/X_train.csv")
X_test = pd.read_csv("/home/mec68/DSG_data/X_test.csv")

X_train.drop(['TodayDate', 'ReceivedDateTime'], inplace=True, axis=1)
X_test.drop(['TodayDate', 'ReceivedDateTime'], inplace=True, axis=1)

Y_train = pd.read_csv("/home/mec68/DSG_data/Y_train.csv")

X_train.insert(0, "Converted", Y_train["Converted"])

# get the snippet
#X_train = X_train.iloc[1:1000][0:]

# hashing the user ids
X_train['CustomerMD5Key'] = pd.factorize(X_train['CustomerMD5Key'])[0]
X_train.rename(columns={'CustomerMD5Key': 'user'}, inplace=True)
X_test.rename(columns={'CustomerMD5Key': 'user'}, inplace=True)

# drop missing values - - only 44 rows!
X_train.isnull().values.any()
X_train.isnull().sum()
X_train = X_train.dropna()

# unnecessary columns
X_train = X_train.drop(['Unnamed: 0'], axis=1)
Y_train = Y_train.drop(['Unnamed: 0'], axis=1)

features = set(X_train.columns)-set(['Converted'])


for f in features:
    if isinstance(X_train.loc[0, f], str) or (len(X_train.loc[:, f].unique()) < 100):
        X_train.loc[:, f] = pd.factorize(X_train[f])[0]

for f in features:
    if isinstance(X_test.loc[0, f], str) or (len(X_test.loc[:, f].unique()) < 100):
        X_test.loc[:, f] = pd.factorize(X_test[f])[0]

#sys.exit()

# Unsupervised clustering of the training data
# getting rid of users features

cols = X_train.columns
#userFeatures = cols - ['Converted', 'user']
carClusters = ['CarFuelId', 'NameOfPolicyProduct', 'CarTransmissionId']
#X_train = X_train.drop(userFeatures, axis=1)
batch_size = 300000
n_clusters = 8
#
# from sklearn.cluster import KMeans
# k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
#
# k_means.fit(X_train)

from sklearn.cluster import MiniBatchKMeans
mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size,
                      n_init=10, max_no_improvement=10, verbose=0)

mbk.fit(X_train)

# results
mbk.cluster_centers_
mbk.labels_

# Predict cluster group of unseen test data
train_clusters = mbk.predict(X_train)
test_clusters = mbk.predict(X_test)

# adding ONE new feature - which cluster
clusterName = 'userCluster'
X_train[clusterName] = train_clusters
X_test[clusterName] = test_clusters

X_train.groupby('user').get_group(1)
X_train.groupby('user').groupby(clusterName)
s = X_train[clusterName].value_counts()

names = ['Clust' + str(ii) for ii in range(n_clusters)]
s = pd.value_counts(X_train[clusterName], sort=False)
df = pd.DataFrame(s, columns=names)
df.loc[0] = s.values

for ii in X_train['user'].unique(): # going through users
    values = [0. for ii in range(len(s))]
    #s = X_train[X_train['user'] == ii][clusterName].value_counts()

    s = pd.value_counts(X_train[X_train['user'] == ii][clusterName])
    df = pd.DataFrame(s, columns=names)
    df.loc[0] = s.values
    #freq = s / sum(s) add frequencies?
    len(s)

X_train.loc[X_train['user'] == 5, clusterName].value_counts()

# adding information about all quests
for ii in n_clusters:
    X_train[str(ii)] =
    X_test[str(ii)]