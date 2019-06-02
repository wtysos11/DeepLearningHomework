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
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

import time

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
dataSet = dataSet[:100000]
labelSet = labelSet[:100000]
'''
X_train, X_test, y_train, y_test = train_test_split(dataSet, labelSet, test_size = 0.05, random_state = 0)

begin = time.time()
regressor = RandomForestRegressor(n_estimators=200,max_depth=9,max_features=5,min_samples_split = 10000,min_samples_leaf = 3000,random_state=10,n_jobs = 4)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
r2_score(y_pred,y_test)
end=time.time()
print(end-begin)
'''
param_test1 = {'min_samples_split':range(80,150,20), 'min_samples_leaf':range(10,60,10)}
gsearch1 = GridSearchCV(estimator=RandomForestRegressor(n_estimators=160,max_depth=9,max_features=5,random_state=10),
param_grid=param_test1,scoring='r2',cv=3,n_jobs=4)
gsearch1.fit(X_train,y_train.ravel())
print(gsearch1.best_params_,gsearch1.best_score_)
'''
'''
#决策树算法时，最大深度为5-8.
#最大特征数量
#决策树数量

决策树数量：100，最大深度：10 r2_score = -3.375
决策树：1000 最大深度：10 max_features = log2 r2_score = -4.74

regressor = RandomForestRegressor(n_estimators = 100,n_jobs = -1)
'''
test = []
for i in range(1,7):
    testSet = pd.read_csv('test'+str(i)+'.csv',header=None)  
    test.append(testSet)

testSet = pd.concat(test)

y_pred = regressor.predict(testSet)

ans = pd.DataFrame(y_pred)
ans.columns = ['Predicted']
ans.index = ans.index+1
ans.index.name = 'id'
ans.to_csv('ans.csv')
