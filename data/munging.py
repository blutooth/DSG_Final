import numpy as np
import pandas as pd

def transform(train):
    train['Gender'] = train.Sex.map({'female': 0, 'male': 1}).astype('int')
    median_ages = np.zeros((2, 3))
    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i, j] = train[(train['Gender'] == i) & \
                                      (train['Pclass'] == j + 1)]['Age'].dropna().median()

    train['AgeFill'] = train['Age']

    for i in range(0, 2):
        for j in range(0, 3):
            train.loc[(train.Age.isnull()) & (train['Gender'] == i) & \
                      (train['Pclass'] == j + 1), 'AgeFill'] = median_ages[i, j]

    train['AgeIsNull'] = pd.isnull(train.Age).astype(int)
    train['FamilySize'] = train['SibSp'] + train['Parch']
    train = train.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)

    return train