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
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

train_set = pd.read_csv('trainSet.csv')
last_test_set = pd.read_csv('test set.csv')


input_size = 32
num_classes = 2
learning_rate = 0.01
num_epochs = 100
model = nn.Linear(input_size, num_classes)

criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

line_name = []
for i in range(1,33):
    line_name.append('F'+str(i))

train_data,test_data,train_target,test_target = train_test_split(train_set[line_name],train_set["label"],test_size=0.2)
ss=StandardScaler()
train_data = ss.fit_transform(train_data)
test_data = ss.transform(test_data)

batch_size = 8000
total_num = int(train_data.shape[0]/batch_size)
current_num = 0
out = False
accu = []
#连续十次没有超过最大值，则取消训练
max_correct = 0
max_iter = 0
max_record = 0
while not out and current_num<batch_size:
    current_num +=1
    for i in range(total_num):
        low = i*batch_size
        high = (i+1)*batch_size
        images = train_data[low:high,:]
        label = train_target.values[low:high]
        outputs = model(torch.from_numpy(images).float())
        outputs = outputs.reshape(high-low,2)
        ans = torch.tensor(label)
        loss = criterion(outputs,ans)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        '''
        if (i+1) % 100 == 0:
            print(current_num,i,loss.item())
        '''
    with torch.no_grad():
        total_num = int(test_data.shape[0]/batch_size)
        correct = 0
        total = 0
        for i in range(total_num):
            low = i*batch_size
            high = (i+1)*batch_size
            images = test_data[low:high,:]
            label = test_target.values[low:high]
            outputs = model(torch.from_numpy(images).float())
            outputs = outputs.reshape(high-low,2)
            _, predicted = torch.max(outputs.data, 1)
            label = torch.from_numpy(label)
            total += label.size(0)
            correct += (predicted == label).sum()
    
        print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
        print(current_num,correct,total)
        accu.append(correct)
        if correct > max_correct:
            max_correct = correct
            max_iter = 0
            max_record = current_num
        else:
            max_iter +=1

        if max_iter > 3000:
            out = True

print(max_record,max_correct)
with torch.no_grad():
    total_num = int(test_data.shape[0]/batch_size)
    correct = 0
    total = 0
    for i in range(total_num):
        low = i*batch_size
        high = (i+1)*batch_size
        images = test_data[low:high,:]
        label = test_target.values[low:high]
        outputs = model(torch.from_numpy(images).float())
        outputs = outputs.reshape(high-low,2)
        _, predicted = torch.max(outputs.data, 1)
        label = torch.from_numpy(label)
        total += label.size(0)
        correct += (predicted == label).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


test_data = ss.transform(last_test_set)
with torch.no_grad():
    images = test_data
    outputs = model(torch.from_numpy(images).float())
    _, predicted = torch.max(outputs.data, 1)

lr_y_predict=predicted

ans = np.zeros((2,183267))
for i in range(183267):
    ans[0][i] = int(i+1)
    ans[1][i] = int(lr_y_predict[i])

output = pd.DataFrame(data=ans.T,index=range(1,183268),columns=["id","Predicted"],dtype=np.int32)
output.to_csv("ans.csv",index=False)
