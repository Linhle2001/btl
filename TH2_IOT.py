# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 01:55:32 2022

@author: Admin
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
import seaborn as sns
#import datasets
dataset = load_boston()

print(dataset.data)
print(dataset.feature_names)
print(dataset.DESCR)
print(dataset.target)

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['MEDV'] = dataset.target
df.head()

#df.info()
#print(df.isnull().sum())

#linear regression

corr = df.corr()
print(corr)
#---get the top 3 features that has the highest correlation---
print(df.corr().abs().nlargest(3, 'MEDV').index)
#---print the top 3 correlation values---
print(df.corr().abs().nlargest(3, 'MEDV').values[:,13])
#plotting heatmap for correlation:
g=sns.heatmap(df.corr(),annot=True,cmap="RdYlGn")

#multiple regression
#xem mối quan hệ giữa LSTAT và MEDV:

plt.scatter(df['LSTAT'], df['MEDV'], marker='o')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.show()

#mối quan hệ giữa RM và MEDV:
plt.scatter(df['RM'], df['MEDV'], marker='o')
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.show()


# xem mối quan hệ giữa RM, LSTAT, MEDV trong biểu đồ 3D:
#from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(18,15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['LSTAT'],
 df['RM'],
 df['MEDV'],
 c='b')
ax.set_xlabel("LSTAT")
ax.set_ylabel("RM")
ax.set_zlabel("MEDV")
plt.show()

#Training model using all featrues:
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
X = df.iloc[:,0:13]  #independent columns
Y = df.iloc[:,-1]    #target column i.e price range
from sklearn.model_selection import train_test_split
x_train, x_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3,random_state=5)
model = LinearRegression()
model.fit(x_train, Y_train)
predicted_y = model.predict(x_test)
result = pd.DataFrame({'Actual:':Y_test, 'Predict:': predicted_y})
# Calculation of Mean Squared Error (MSE)
mse = mean_squared_error(Y_test, predicted_y)
#calculate R-squared of regression model
r_squared = model.score(X, Y)
print('Using all features:')
print(result)
print('R-Squared: %.4f'%r_squared)
print('MSE: ')
print(mse)

#Traning model with 2 features have the highest corr:
x = pd.DataFrame(np.c_[df['LSTAT'], df['RM']], columns = ['LSTAT','RM'])
Y = df['MEDV']
#70 percent for training and 30 percent for testing:
from sklearn.model_selection import train_test_split
x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size = 0.3,
 random_state=5)
#training sets:
print(x_train.shape)
print(Y_train.shape)
#testing sets:
print(x_train.shape)
print(Y_train.shape)


model = LinearRegression()
model.fit(x_train, Y_train)
price_pred = model.predict(x_test)
result = pd.DataFrame({'Actual:':Y_test, 'Predict:': price_pred})
print('Using 2 featrue have the highest corr:')
print('R-Squared: %.4f' % model.score(x_test,
 Y_test))
mse = mean_squared_error(Y_test, price_pred)
print('MSE:')
print(mse)

#plotting to see the difference between Actual and Predict
plt.scatter(Y_test, price_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted prices")
plt.title("Actual prices vs Predicted prices")
#Predict Price with LSTAT=30 and RM=5
print(model.predict([[30,5]]))

#Plotting the 3D Hyperplane:

from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_boston
dataset = load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['MEDV'] = dataset.target
x = pd.DataFrame(np.c_[df['LSTAT'], df['RM']], columns = ['LSTAT','RM'])
Y = df['MEDV']
fig = plt.figure(figsize=(18,15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x['LSTAT'],
 x['RM'],
 Y,
 c='b')
ax.set_xlabel("LSTAT")
ax.set_ylabel("RM")
ax.set_zlabel("MEDV")
#---create a meshgrid of all the values for LSTAT and RM---
x_surf = np.arange(0, 40, 1) #---for LSTAT---
y_surf = np.arange(0, 10, 1) #---for RM---
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x, Y)
#---calculate z(MEDC) based on the model---
z = lambda x,y: (model.intercept_ + model.coef_[0] * x + model.coef_[1] * y)
ax.plot_surface(x_surf, y_surf, z(x_surf,y_surf),
 rstride=1,
 cstride=1,
 color='None',
 alpha = 0.4)
plt.show()

