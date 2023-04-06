import numpy as np
import pandas as pd
import MySQLdb.connections

mydb = MySQLdb.Connect(
  host="localhost",
  user="root",
  password="",
  database="iot"
)

mycursor = mydb.cursor()

mycursor.execute("SELECT * FROM csvimport limit 10")

myresult = mycursor.fetchall()
df = pd.read_sql("select*from csvimport where Humidity<48", mydb)
df = df[['Humidity', 'Temperature', 'Raw_Ethanol', 'Pressure', 'Fire_Alarm']]

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # for leaf node
        self.value = value
class DecisionTreeClassifierBuild():
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree ''' 
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        # return best split
        return best_split
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''
        
        weight_left = len(l_child) / len(parent)
        weight_right = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_left*self.gini_index(l_child) + weight_right*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_left*self.entropy(l_child) + weight_right*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        ''' function to compute entropy '''
        
        labels = np.unique(y)
        entropy = 0
        for i in labels:
            p = len(y[y == i]) / len(y)
            entropy += -p * np.log2(p)
        return entropy
    
    def gini_index(self, y):
        ''' function to compute gini index '''
        
        labels = np.unique(y)
        gini = 0
        for i in labels:
            p = len(y[y == i]) / len(y)
            gini += p**2
        return 1 - gini
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        ''' function to train the tree '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        ''' function to predict new dataset '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    def make_prediction(self, x, tree):
       ''' function to predict a single data point '''
       
       if tree.value!=None: return tree.value
       feature_val = x[tree.feature_index]
       if feature_val<=tree.threshold:
           return self.make_prediction(x, tree.left)
       else:
           return self.make_prediction(x, tree.right)
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)
#Fitting sử dụng hàm tự xây dựng
classifier = DecisionTreeClassifierBuild(min_samples_split=3, max_depth=3)
classifier.fit(X_train,Y_train)
classifier.print_tree()
Y_pred = classifier.predict(X_test) 

#Fitting Decision Tree classifier sử dụng sklearn  
from sklearn.tree import DecisionTreeClassifier  
classifier2= DecisionTreeClassifier(criterion='entropy', random_state=0)  
classifier2.fit(X_train, Y_train) 
pred = classifier2.predict(X_test)

Y_pred = np.asarray(Y_pred)
# Hàm kết nối
def connect():
    """ Kết nối MySQL bằng module MySQLConnection """
   
    # Biến lưu trữ kết nối
    conn = None
 
    try:
        conn = MySQLdb.Connect(
          host="localhost",
          user="root",
          password="",
          database="iot"
        )
 
    except :
        print("error")
 
    return conn
def insert_predict(skl, build):
    query = "INSERT INTO prediction(sklearn,build) " \
            "VALUES(%s,%s)"
    args = (skl, build)
 
    try:
 
        conn = connect()
 
        cursor = conn.cursor()
        cursor.execute(query, args)

        conn.commit()
    
    finally:
        # Đóng kết nối
        cursor.close()
        conn.close()
'''
for x in range(pred.size):
    mycursor.execute("insert into prediction(sklearn, build) values(%s,%s)", (int(pred[x]),int(Y_pred[x])))
    print('ok')
'''
'''

for x in range(pred.size):
    insert_predict(pred[x], Y_pred[x])
    print(int(pred[x]),int(Y_pred[x]))
'''

#Evaluate
def calculate_metrics(predicted, actual):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(predicted)):
        if   (predicted[i] == '1') & (actual[i] == '1'):
            TP += 1
        elif (predicted[i] == '0') & (actual[i] == '1'):
            FP += 1
        elif (predicted[i] == '0') & (actual[i] == '0'):
            TN += 1
        else:
            FN += 1
    
    accuracy  = (TP + TN) / (TP + FP + TN + FN) 
    precision = (TP) / (TP+FP)
    recall    = (TP) / (TP + FN) 
    f1_score  = (2 * precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, f1_score
print(calculate_metrics(pred, Y_test))

print(type(type(pred[1])))
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print(accuracy_score(Y_test, pred))
print(recall_score(Y_test, pred,pos_label='1'))
print(f1_score(Y_test, pred,pos_label='1'))