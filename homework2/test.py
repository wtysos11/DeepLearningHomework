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

dataSet = pd.read_csv('train1.csv',header=None)
ansSet = pd.read_csv('label1.csv',header=None)

X_train, X_test, y_train, y_test = train_test_split(dataSet, ansSet, test_size = 0.2, random_state = 0)

ssX = StandardScaler()
X_train = ssX.fit_transform(X_train)
X_test = ssX.transform(X_test)
ssY = StandardScaler()
y_train = ssX.fit_transform(y_train)
y_test = ssX.transform(y_test)

import time

begin = time.time()
regressor = RandomForestRegressor(n_estimators = 50,random_state = 0,n_jobs = -1)
regressor.fit(X_train, y_train.ravel())
end = time.time()
print(end-begin)

y_pred = regressor.predict(X_test)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred)