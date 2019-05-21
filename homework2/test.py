import os

os.chdir('E:\\code\\homework\\deepLearning\\homework2')

#pandas用于处理数据
import pandas as pd                                                         
import numpy as np
#用于分析数据                 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

data = []
label = []

for i in range(1,6):
    dataSet = pd.read_csv('train'+str(i)+'.csv',header=None)  
    ansSet = pd.read_csv('label'+str(i)+'.csv',header=None)  
    data.append(dataSet)
    label.append(ansSet)

dataSet = pd.concat(data)
labelSet = pd.concat(label)

'''
dataSet = pd.read_csv('train1.csv',header=None)
labelSet = pd.read_csv('label1.csv',header=None)
'''

X_train = dataSet
y_train = labelSet
#X_train, X_test, y_train, y_test = train_test_split(dataSet, labelSet, test_size = 0.05, random_state = 0)

ssX = StandardScaler()
X_train = ssX.fit_transform(X_train)

ssY = StandardScaler()
y_train = ssX.fit_transform(y_train)


import time

begin = time.time()
regressor = RandomForestRegressor(n_estimators = 200,max_features = 5,max_depth = 10,random_state = 0,n_jobs = 4,oob_score = True)
regressor.fit(X_train, y_train.ravel())
end = time.time()
print(end-begin)

test = []
for i in range(1,7):
    testSet = pd.read_csv('test'+str(i)+'.csv',header=None)  
    test.append(testSet)

testSet = pd.concat(test)

X_test = ssX.transform(testSet)

y_pred = regressor.predict(testSet)

ans = pd.DataFrame(y_pred)
ans.columns = ['Predicted']
ans.index = ans.index+1
ans.index.name = 'id'
ans.to_csv('ans.csv')