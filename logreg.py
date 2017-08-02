# standard imports to help with visualization 
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm

def logistic(t):
    return 1.0 / (1 + math.exp((-1.0)*t))
t = np.linspace(-4,4,500)

y = np.array([logistic(val) for val in t])

plt.plot(t,y)
plt.title("Log Function")
#plt.show()

df = sm.datasets.fair.load_pandas().data

def affair_check(x):
    if x != 0:
        return 1
    else:
        return 0

df['Had_Affair'] = df['affairs'].apply(affair_check)

df.groupby('Had_Affair').mean()


# visualize the data
sns.countplot('age',data=df,hue='Had_Affair',palette='coolwarm')
sns.plt.show()

sns.countplot('yrs_married',data=df,hue='Had_Affair',palette='coolwarm')
sns.plt.show()

sns.countplot('educ',data=df,hue='Had_Affair',palette='coolwarm')
sns.plt.show()

sns.countplot('religious',data=df,hue='Had_Affair',palette='coolwarm')
sns.plt.show()

occ_d = pd.get_dummies(df['occupation'])
hus_occ_d = pd.get_dummies(df['occupation_husb'])

occ_d.columns = ['occ1','occ2','occ3','occ4','occ5','occ6']
hus_occ_d.columns = ['hocc1','hocc2','hocc3','hocc4','hocc5','hocc6']

X = df.drop(['occupation','occupation_husb','Had_Affair'],axis=1)

dummies = pd.concat([occ_d,hus_occ_d],axis=1)

X = pd.concat([X,dummies],axis=1)
Y = df.Had_Affair

X = X.drop('occ1',axis=1)
X = X.drop('hocc1',axis=1)
#this is a repeat of our target data, we should get get rid of it
X = X.drop('affairs',axis=1)

Y = np.ravel(Y)

log_model = LogisticRegression()
log_model.fit(X,Y)
print(log_model.score(X,Y))

Y.mean()
#null error rate is 1 - Y.mean() => 0.68 68% accuracy

# post = decrease of affair
coeff_df = DataFrame(zip(X.columns,np.transpose(log_model.coef_)))

X_train, X_test, Y_train,Y_test = sklearn.model_selection.train_test_split(X,Y)

log_model2 = LogisticRegression()
log_model2.fit(X_train,Y_train)

class_predict = log_model2.predict(X_test)

print metrics.accuracy_score(Y_test,class_predict)


