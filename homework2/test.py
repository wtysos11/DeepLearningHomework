import os

os.chdir('E:\\code\\homework\\deepLearning\\homework2')

#pandas用于处理数据
import pandas as pd                                                         
import numpy as np
#用于分析数据                 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestRegressor
from tree import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from multiprocessing import freeze_support
import time



'''
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

#进行并行化测试
if __name__ == "__main__":
    freeze_support()
    dataSet = pd.read_csv('train1.csv',header=None)
    labelSet = pd.read_csv('label1.csv',header=None)
    dataSet = dataSet[:100000]
    labelSet = labelSet[:100000]       
    
    

    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(dataSet, labelSet, test_size = 0.05, random_state = 0)
        begin = time.time()
        regressor = RandomForestRegressor(n_estimators=5,max_depth=3,n_jobs = 0)
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
'''

'''
使用原始的5个dataSet合并，构成极大pandas.DataFrame

生成极大矩阵随机索引：dataRow

场景1：直接使用dataRow来访问dataFrame
场景2：保存dataFrame.values，使用numpy数组来访问。
场景3：使用numpy数组对于dataRow的拷贝来访问

对于以上3个场景，考虑遍历加和的时间，重复3次求平均值
'''
