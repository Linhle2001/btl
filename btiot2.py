# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 14:35:19 2022

@author: Admin
"""

from sklearn.datasets import load_diabetes
import pandas as pd
import matplotlib as plt
import seaborn as sns
data = load_diabetes()
df = pd.DataFrame(data.data, columns= data.feature_names)
df['Y'] = data.target
X = df.iloc[:,0:10]  #independent columns
Y = df.iloc[:,-1]    #target column i.e price range
#get correlations of each features in dataset
corr = df.corr()
g=sns.heatmap(df.corr(),annot=True,cmap="RdYlGn")
from sklearn.model_selection import train_test_split
x_train, x_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3,random_state=5)
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, Y_train)
predicted_y = model.predict(x_test)
result = pd.DataFrame({'Actual:':Y_test, 'Predict:': predicted_y})
# Calculation of Mean Squared Error (MSE)
mse = mean_squared_error(Y_test, predicted_y)
#calculate R-squared of regression model
r_squared = model.score(X, Y)
print('10 features:')
print(result)
print(r_squared)
print(mse)

