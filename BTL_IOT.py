# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 09:28:13 2022

@author: Admin
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

df = pd.read_csv ('smoke_detection_iot.csv')
df.info ()

# --- kiểm tra các giá trị null ---
print ("Nulls")
print ("=====")
print (df.isnull (). sum ())
# --- kiểm tra 0s ---
print ("0s")
print ("==")
print (df.eq (0) .sum ())
'''
df.eq(0).sum().reset_index(name="null").plot.bar(x='index', y='null', rot=90)
def find_missing_values(dataframe, columns):
    missing_values = {}
    df_rows = len(dataframe)
    for column in columns:
        tot_column_values = dataframe[column].value_counts().sum()
        missing_values[column] = df_rows - tot_column_values
    return missing_values
missing_values = find_missing_values(df, columns = df.columns)
missing_values
df.isna().sum().to_dict()

'''
df.drop(['Unnamed: 0','UTC'], axis = 1, inplace=True) #Removing Unnecessary Columns
'''
#Checking If Dataset is Imbalanced
from collections import Counter
c = Counter(df['Fire Alarm'])

fig = px.bar(df ,x=[i[0] for i in c.items()],y =[i[1] for i in c.items()], color=[i[1] for i in c.items()], template = 'plotly_dark',height = 500,width = 700)
fig.update_layout(xaxis_title = 'Fire Alarm', yaxis_title = 'Count',template = 'plotly_dark',title = 'Fire Alarm Distribution')
fig.show()
'''
#Cleaning the column names (Removing '[]')
for i in df.columns:
    if '[' in i:
        df.rename(columns = {i:i[:i.index('[')]},inplace=True)
#print(df.head())

#Checking for outliers
fig, ax = plt.subplots(figsize = (13,8))
plt.xticks(rotation = 90)
sns.boxplot(data = df, ax = ax, palette='Reds')

plt.figure(figsize = (13,8))

#Select features
sns.heatmap(df.corr(), annot = True, cmap= 'Blues')
print(df.corr().abs().nlargest(5, 'Fire Alarm').index)# CNT, Humidity, Raw Ethanol, Pressure
#Splitting data
'''
#Using all features
X = df.iloc[:,0:13]  #independent columns
Y = df.iloc[:,-1]    #target column 
x_train, x_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3,random_state=5)

#Using logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
log_regress = LogisticRegression()
log_regress_score = cross_val_score(log_regress, X, Y, cv=10, scoring='accuracy').mean()
print(log_regress_score)
model = LogisticRegression()
model.fit(x_train, Y_train)
predicted_y = model.predict(x_test)
result = pd.DataFrame({'Actual:':Y_test, 'Predict:': predicted_y})
print(result)
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predicted_y))
'''
'''
#Using KNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
#---empty list that will hold cv (cross-validates) scores---
cv_scores = []
#---number of folds---
folds = 5
#---creating odd list of K for KNN---
ks = list(range(1,int(len(X) * ((folds - 1)/folds)), 2))
#---perform k-fold cross validation---
for k in ks:
 knn = KNeighborsClassifier(n_neighbors=k)
 score = cross_val_score(knn, X, Y, cv=folds, scoring='accuracy').mean()
 cv_scores.append(score)
#---get the maximum score---
knn_score = max(cv_scores)
#---find the optimal k that gives the highest score---
optimal_k = ks[cv_scores.index(knn_score)]
print(f"The optimal number of neighbors is {optimal_k}")
print(knn_score)
'''
'''
#Using SVM
from sklearn.model_selection import cross_val_score
from sklearn import svm
linear_svm = svm.SVC(kernel='linear')
linear_svm_score = cross_val_score(linear_svm, X, Y, cv=10, scoring='accuracy').mean()
print(linear_svm_score)
linear_svm.fit(x_train, Y_train)
predicted_y = linear_svm.predict(x_test)
result = pd.DataFrame({'Actual:':Y_test, 'Predict:': predicted_y})
print(result)
'''

#Using 4 features have highest score:
X = df[['Humidity','Raw Ethanol', 'Pressure', 'Temperature']] #independent columns
Y = df.iloc[:,-1]    #target column 
x_train, x_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3,random_state=5)

#Using logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
log_regress = LogisticRegression()
log_regress_score = cross_val_score(log_regress, X, Y, cv=10, scoring='accuracy').mean()
print(log_regress_score)
model = LogisticRegression()
model.fit(x_train, Y_train)
predicted_y = model.predict(x_test)
result = pd.DataFrame({'Actual:':Y_test, 'Predict:': predicted_y})
print(result)
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predicted_y))
print(model.predict([[19701,939.784,13,56.77]]))
from sqlalchemy import create_engine

import pymysql

import pandas as pd

tableName   = "csv"
         

 

sqlEngine       = create_engine('mysql+pymysql://root:@127.0.0.1/iot', pool_recycle=3600)

dbConnection    = sqlEngine.connect()

 

try:

    frame           = df.to_sql(tableName, dbConnection, if_exists='fail');

except ValueError as vx:

    print(vx)

except Exception as ex:   

    print(ex)

else:

    print("Table %s created successfully."%tableName);   

finally:

    dbConnection.close()