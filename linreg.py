import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.datasets import load_boston

boston = load_boston()

plt.hist(boston.target,bins=50)
plt.xlabel('Prices in thousands')
plt.ylabel('Number of Houses')
plt.show()

plt.scatter(boston.data[:,5],boston.target)
plt.ylabel('Prices in thousands')
plt.xlabel('Number of Rooms')
plt.show()

boston_df = DataFrame(boston.data)
boston_df.columns = boston.feature_names
boston_df['Price'] = boston.target

#Linear fit
sns.lmplot('RM','Price',data=boston_df)
sns.plt.show()

X = boston_df.RM
# numpy needs to know how many values and atributes
X = np.vstack(boston_df.RM)
Y = boston_df.Price

# want to create a matrix of form [x 1]
X = np.array([[value,1] for value in X])

m , b = np.linalg.lstsq(X,Y)[0]

plt.plot(boston_df.RM,boston_df.Price,'o')
x = boston_df.RM
plt.plot(x, m*x + b,'r',label= 'Best Fit Line')
plt.show()

result = np.linalg.lstsq(X,Y)

error_total = result[1]

rmse = np.sqrt(error_total)/ len(X)

print(str(rmse))


#Now for a multivariate regression

import sklearn
from sklearn.linear_model import LinearRegression

lreg = LinearRegression()
# An R^2 of 1 indicates perfect linear fitting

X_multi = boston_df.drop('Price',axis=1)
Y_target = boston_df.Price

lreg.fit(X_multi,Y_target)
print(' The estimated intercept coeff is %.2f' %lreg.intercept_)
print(' The number of coeff used was %d ' %len(lreg.coef_))

coeff_df = DataFrame(boston_df.columns)
coeff_df.columns = ['Features']
coeff_df['Coefficient Estimate'] = Series(lreg.coef_)

# using training and validation

X_train, X_test, Y_train,Y_test = sklearn.model_selection.train_test_split(X,boston_df.Price)

print(X_train.shape, X_test.shape, Y_train.shape,Y_test.shape)

lreg = LinearRegression()
lreg.fit(X_train,Y_train)

pred_train = lreg.predict(X_train)
pred_test = lreg.predict(X_test)

print("Fit a model X_train and calc the MSE with Y_train: %.2f" % np.mean((Y_train - pred_train)**2))
print("Fit a model X_train and calc the MSE with X_test and Y_test: %.2f" % np.mean((Y_test - pred_test)**2))

#residual plots is the difference between the observed value - predicted value

train = plt.scatter(pred_train,(pred_train - Y_train),c='teal',alpha=0.5)
test = plt.scatter(pred_test,(pred_test - Y_test),c='maroon',alpha=0.5)
plt.hlines(y=0,xmin=-10,xmax=50)
plt.legend((train,test),('Training','Test'), loc = 'lower left')
plt.title('Residual Plots')
plt.show()







