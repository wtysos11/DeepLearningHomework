import os

os.chdir('E:\\code\\homework\\deepLearning\\homework1')

import pandas as pd                                                         
import numpy as np                                       
from sklearn.preprocessing import StandardScaler        
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import SGDClassifier                   
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import time

train_set = pd.read_csv('trainSet.csv')
last_test_set = pd.read_csv('test set.csv')

line_name = []
for i in range(1,33):
    line_name.append('F'+str(i))


train_data,test_data,train_target,test_target = train_test_split(train_set[line_name],train_set["label"],test_size=0.2)
ss=StandardScaler()
train_data = ss.fit_transform(train_data)
test_data = ss.transform(test_data)
begin = time.time()
lgReg=LogisticRegression(penalty='l2',solver='lbfgs',multi_class="ovr",max_iter = 1000,n_jobs=-1) #生成逻辑回归器
lgReg.fit(train_data,train_target) #进行训练
end = time.time()
ans_target = lgReg.predict(test_data) #产生测试数据
print(accuracy_score(test_target,ans_target))

#逻辑：1.k-fold，分成5组，其中选4组作为训练集，1组作为测试集，经过测试后准确率最高的作为最终结果
#2. 初始权值为-1~1中的随机数，每次更新一次权值

test_data = ss.transform(last_test_set)
lr_y_predict=lgReg.predict(test_data)

ans = np.zeros((2,183267))
for i in range(183267):
    ans[0][i] = int(i+1)
    ans[1][i] = int(lr_y_predict[i])

output = pd.DataFrame(data=ans.T,index=range(1,183268),columns=["id","Predicted"],dtype=np.int32)
output.to_csv("ans.csv",index=False)

