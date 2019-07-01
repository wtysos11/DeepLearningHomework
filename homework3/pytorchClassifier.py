import os
os.chdir('E:\\code\\homework\\deepLearning\\homework3')
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchstat import stat
from torch.autograd import Variable
epoch_times = 3
batch_size = 4
split_count = 15000

labels = pd.read_csv("train label.csv")
imgsNames = labels.values[:,0]
labelsNum = labels.values[:,1]
num_classes = len(set(labelsNum))

train_dir = 'Cloth image Dataset/image/train/'
file = imgsNames
file = [train_dir+i for i in file ]
number = labelsNum
file_train,file_test,number_train,number_test = train_test_split(file,number,test_size = 0.05)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    #transforms.Scale(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

def default_loader(path):
    img_pil =  Image.open(path)
    img_pil = img_pil.convert('RGB')
    img_pil = img_pil.resize((350,350))
    img_tensor = preprocess(img_pil)
    return img_tensor

#当然出来的时候已经全都变成了tensor
class trainset(Dataset):
    def __init__(self, loader=default_loader):
        #定义好 image 的路径
        self.images = file_train
        self.target = number_train
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)
    
class testset(Dataset):
    def __init__(self, loader=default_loader):
        #定义好 image 的路径
        self.images = file_test
        self.target = number_test
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)

import torch
import torch.nn as nn
from torchvision.models import resnet152
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.model = resnet152(True)
        self.fc = nn.Linear(1000,8)
    
    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)

        #x = self.model.classifier(self.features)
        return x

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return res

'''
import torch
import torch.nn as nn
from torchvision.models import resnet18

last_channel = 512
class LeNet(nn.Module):
 
    def __init__(self):
        #Net继承nn.Module类，这里初始化调用Module中的一些方法和属性
        nn.Module.__init__(self) 
 
        #定义特征工程网络层，用于从输入数据中进行抽象提取特征
        self.feature_engineering = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=3),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3),
            nn.MaxPool2d(kernel_size=2,
                        stride=2),
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3),
            nn.MaxPool2d(kernel_size=2,
                        stride=2),
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=3),
            nn.Conv2d(in_channels=512,
                      out_channels=last_channel,
                      kernel_size=3),
        )
        #分类器层，将self.feature_engineering中的输出的数据进行拟合
        self.classifier = nn.Sequential(
            nn.Linear(in_features=last_channel*36*36,
                      out_features=120),
            nn.Linear(in_features=120,
                      out_features=84),
            nn.Linear(in_features=84,
                      out_features=num_classes),
        )
 
    def forward(self, x):
        #在Net中改写nn.Module中的forward方法。
        #这里定义的forward不是调用，我们可以理解成数据流的方向，给net输入数据inpput会按照forward提示的流程进行处理和操作并输出数据
        x = self.feature_engineering(x)
        x = x.view(-1, last_channel*36*36)
        x = self.classifier(x)
        return x
'''
print(1)

train_data  = trainset()
trainloader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
test_data = testset()
testloader = DataLoader(test_data, batch_size=batch_size,shuffle=True)

#del(net)
device = torch.device("cuda:0")
net = LeNet().to(device)

#定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()
 
#随机梯度下降SGD优化器
optimizer = optim.SGD(params = net.parameters(),lr = 0.001)

predict_dir = 'Cloth image Dataset/image/test/'
test_imgs = os.listdir(predict_dir)
file = test_imgs
file_predict = [predict_dir+i for i in file ]

class predictset(Dataset):
    def __init__(self, loader=default_loader):
        #定义好 image 的路径
        self.images = file_predict
        self.names = test_imgs
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        name = self.names[index]
        return (img,name)

    def __len__(self):
        return len(self.images)


def getValidate():
    test_data = testset()
    testloader = DataLoader(test_data, batch_size=batch_size,shuffle=True)
    correct = 0
    total = 0
    average = 0
    with torch.no_grad():
        for data in testloader:
            images,labels = data
            images, labels = images.to(device), labels.to(device)
            #outputs = net(inputs)
            outputs = net(images)#resnet
            loss = criterion(outputs,labels)
            average += loss
            ans = torch.max(outputs,dim=1).indices
            total += labels.size(0)
            correct += (ans == labels).sum().item()
            del(images)
            del(labels)
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    return loss/total
import time
s = time.time()
record = []
aver_loss = []
for epoch in range(epoch_times):
    for i,data in enumerate(trainloader):
        torch.cuda.empty_cache()
        inputs,labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        #outputs = net(inputs)
        outputs = net(inputs)#resnet
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        del(inputs)
        del(labels)
        
        if i%100 == 0:
            c = time.time()
            print('[{}/{}]loss:{}:time:{}s'.format(epoch,i,loss.item(),c-s))
            record.append(loss.item())
            s=c
    print("epoch{}".format(epoch_times))
    ave = getValidate()
    aver_loss.append(ave)
    
plt.plot(record)
plt.show()
plt.plot(aver_loss)
plt.show()

torch.cuda.empty_cache()
predict_data = predictset()
predictloader = DataLoader(predict_data, batch_size=batch_size,shuffle=True)
y_result = []
names = []
with torch.no_grad():
    for data in predictloader:
        images,name = data
        images= images.to(device)
        #outputs = net(inputs)
        outputs = net.forward_classifier(images)#resnet
        ans = torch.max(outputs,dim=1).indices
        y_result += ans.tolist()
        names += list(name)
        del(images)

y = pd.DataFrame(y_result,names)
y.columns = ['Cloth_label']
y.index.name = 'Image'
y.to_csv('ans.csv')