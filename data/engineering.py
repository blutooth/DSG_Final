# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# get titanic & test csv files as a DataFrame
X_train = pd.read_csv("/home/mec68/DSG_data/X_train.csv")
X_test = pd.read_csv("/home/mec68/DSG_data/X_test.csv")
Y_train = pd.read_csv("/home/mec68/DSG_data/Y_train.csv")

# preview the data
titanic_df.head()
titanic_df.info()
titanic_df = titanic_df.drop(['PassengerId','Name', 'Ticket'], axis=1)
test_df    = test_df.drop(['Name'], axis=1)
titanic_df = titanic_df.drop(['Cabin','Sex'], axis=1)
titanic_df = titanic_df.dropna()
# how many missing values:
for ii in titanic_df.columns:
    print(ii)
    print(pd.isnull(titanic_df[ii]).sum())

# Embarked
titanic_df.Embarked[pd.isnull(titanic_df.Embarked)] = titanic_df.Embarked.mode()[0]
# or
titanic_df["Embarked"] = titanic_df["Embarked"].fillna(titanic_df.Embarked.mode()[0])

titanic_df['Gender'] = titanic_df.Sex.map({'female': 0, 'male': 1}).astype('int')
# Age
titanic_df.loc[titanic_df.Survived == 1, 'AgeFill'].median()
titanic_df.loc[titanic_df.Survived == 0, 'AgeFill'].median()
titanic_df['Age'].hist(by=titanic_df['Survived'])
titanic_df['Age'].hist(by=titanic_df['Sex'])
titanic_df['Survived'].hist(by=titanic_df['Embarked'])

median_ages = np.zeros((2,3))
for ii in range(2):
    for jj in range(3):
        median_ages[ii, jj] = titanic_df[(titanic_df['Gender'] == ii) \
                                         & (titanic_df['Pclass'] == jj + 1) ]['Age'].median()

titanic_df['AgeFill'] = titanic_df['Age']
for ii in range(2):
    for jj in range(3):
        titanic_df.loc[(titanic_df.Age.isnull()) & (titanic_df['Gender'] == ii) & \
                      (titanic_df['Pclass'] == jj + 1), 'AgeFill'] = median_ages[ii, jj]
titanic_df.Gender
titanic_df['AgeFill']
# Cabin
titanic_df.Cabin.value_counts()

# value counts for categoricals
titanic_df.Embarked.value_counts()
titanic_df.Age.hist()
titanic_df.Embarked[pd.isnull(titanic_df.Embarked)] = titanic_df.Embarked.hist()
titanic_df.columns
titanic_df.Ticket

# Age
# plt.scatter, sns.boxplot,plt.hist
facet = sns.FacetGrid(titanic_df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, titanic_df['Age'].max()))
facet.add_legend()

titanic_df.Fare.hist(by=titanic_df['Survived'])
# pairwise relation with classes!
g = sns.PairGrid(titanic_df, hue="Survived")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter);
g.add_legend();

g = sns.FacetGrid(titanic_df, col="Survived")
g = g.map(sns.boxplot, "Pclass", "AgeFill")

sns.countplot(x='Pclass', data=titanic_df, hue="Survived")


# factorize categorical variables
all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]

print('Eliminate missing values')
all_data.fillna(missing_indicator, inplace=True)
Center, Scale, and Box-Cox transformation on numericals.