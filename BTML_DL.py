# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 22:56:12 2022

@author: Admin
"""

import pandas as pd
import numpy as np
df = pd.read_csv("diemthi.csv")
#print(df.isnull().sum())

df[['10%','20%_1','20%_2','thi']] = \
df[['10%','20%_1','20%_2','thi']].replace(0,np.NaN)

df.fillna(df.mean(), inplace = True)
#print(df.isnull().sum())

corr = df.corr()
print(corr)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
X = df.iloc[:,1:4]  #independent columns -- 228 rows
Y = df.iloc[:,-1]    #target column 
from sklearn.model_selection import train_test_split
x_train, x_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3,\
                                                    random_state=5)
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
print("Nhap diem 10%, 20%, 20%:")
arr = input().split(" ");
prediction = model.predict([arr])

print(prediction)
from joblib import dump
dump(model,"C:/Users/Admin/deployML/savedModel/model.joblib")
