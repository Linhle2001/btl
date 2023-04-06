# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 15:33:03 2022

@author: Admin
"""

import MySQLdb.connections
import pandas as pd
from math import log, e
import numpy as np
mydb = MySQLdb.Connect(
  host="localhost",
  user="root",
  password="",
  database="iot"
)

mycursor = mydb.cursor()

mycursor.execute("SELECT * FROM csvimport limit 10")

myresult = mycursor.fetchall()
df = pd.read_sql("select*from csvimport", mydb)
df = df[['Humidity', 'Temperature', 'Raw_Ethanol', 'Pressure', 'Fire_Alarm']]
print(df.head())
for x in myresult:
  print(x)
def entropy3(labels, base=None):
  vc = pd.Series(labels).value_counts(normalize=True, sort=False)
  base = e if base is None else base
  return -(vc * np.log(vc)/np.log(base)).sum()
'''
class TreeNode(object):
    def __init__(self, ids = None, children = [], entropy = 0, depth = 0):
        self.ids = ids           # index of data in this node
        self.entropy = entropy   # entropy, will fill later
        self.depth = depth       # distance to root node
        self.split_attribute = None # which attribute is chosen, it non-leaf
        self.children = children # list of its child nodes
        self.order = None       # order of values of split_attribute in children
        self.label = None       # label of node if it is a leaf

    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute # split at which attribute
        self.order = order # order of this node's children 

    def set_label(self, label):
        self.label = label # set label if the node is a leaf 
'''
##1. claculate entropy o the whole dataset
eps = np.finfo(float).eps

entropy_node = 0  #Initialize Entropy
values = df['Fire_Alarm'].unique()
for value in values:
    fraction = df['Fire_Alarm'].value_counts()[value]/len(df['Fire_Alarm'])  
    entropy_node += -fraction*np.log2(fraction)

print(f'Values: {values}')
print(f'entropy_node: {entropy_node}')  
def ent(df,attribute):
    target_variables = df['Fire_Alarm'].unique()  #This gives all 'Yes' and 'No'
    variables = df[attribute].unique()    #This gives different features in that attribute (like 'Sweet')


    entropy_attribute = 0
    for variable in variables:
        entropy_each_feature = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute]==variable][df['Fire_Alarm'] ==target_variable]) #numerator
            den = len(df[attribute][df[attribute]==variable])  #denominator
            fraction = num/(den+eps)  #pi
            entropy_each_feature += -fraction*log(fraction+eps) #This calculates entropy for one feature like 'Sweet'
        fraction2 = den/len(df)
        entropy_attribute += -fraction2*entropy_each_feature   #Sums up all the entropy ETaste

    return(abs(entropy_attribute))


a_entropy = {k:ent(df,k) for k in df.keys()[:-1]}
print(a_entropy)
def ig(e_dataset,e_attr):
    return(e_dataset-e_attr)
IG = {k:ig(entropy_node,a_entropy[k]) for k in a_entropy}
def find_entropy(df):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    return entropy
  
  
def find_entropy_attribute(df,attribute):
  Class = df.keys()[-1]   #To make the code generic, changing target variable class name
  target_variables = df[Class].unique()  #This gives all 'Yes' and 'No'
  variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
  entropy2 = 0
  for variable in variables:
      entropy = 0
      for target_variable in target_variables:
          num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
          den = len(df[attribute][df[attribute]==variable])
          fraction = num/(den+eps)
          entropy += -fraction*log(fraction+eps)
      fraction2 = den/len(df)
      entropy2 += -fraction2*entropy
  return abs(entropy2)
def find_winner(df):
    Entropy_att = []
    IG = []
    for key in df.keys()[:-1]:
#         Entropy_att.append(find_entropy_attribute(df,key))
        IG.append(find_entropy(df)-find_entropy_attribute(df,key))
    return df.keys()[:-1][np.argmax(IG)]
  
def get_subtable(df, node,value):
  return df[df[node] == value].reset_index(drop=True)

def buildTree(df,tree=None): 
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    
    #Here we build our decision tree

    #Get attribute with maximum information gain
    node = find_winner(df)
    
    #Get distinct value of that attribute e.g Salary is node and Low,Med and High are values
    attValue = np.unique(df[node])
    
    #Create an empty dictionary to create tree    
    if tree is None:                    
        tree={}
        tree[node] = {}
    
   #We make loop to construct a tree by calling this function recursively. 
    #In this we check if the subset is pure and stops if it is pure. 

    for value in attValue:
        
        subtable = get_subtable(df,node,value)
        clValue,counts = np.unique(subtable[Class],return_counts=True)                        
        
        if len(counts)==1:#Checking purity of subset
            tree[node][value] = clValue[0]                                                    
        else:        
            tree[node][value] = buildTree(subtable) #Calling the function recursively 
                   
    return tree
t  = buildTree(df)
import pprint
pprint.pprint(t)
test={'Humidity':'20','Temperature':'50','Raw_Ethanol':'10','Pressure':'1000'}
def func(test, tree, default=None):
    attribute = next(iter(tree)) 
    print(attribute) 
    if test[attribute] in tree[attribute].keys():
        print(tree[attribute].keys())
        print(test[attribute])
        result = tree[attribute][test[attribute]]
        if isinstance(result, dict):
            return func(test, result)
        else:
            return result
    else:
        return default
ans = func(test, t)
print(ans)
