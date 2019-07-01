import os
os.chdir('E:\\code\\homework\\deepLearning\\homework3')

import pandas as pd
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
from keras import optimizers
from keras import layers,models
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from PIL import Image

num_classes = 8
num_epochs = 2
batch_size = 20

labels = pd.read_csv("train label.csv")
imgsNames = labels.values[:,0]

model = Sequential()
# 第一个卷积层，32个卷积核，大小５x5，卷积模式SAME,激活函数relu,输入张量的大小
model.add(Conv2D(filters= 64, kernel_size=(3,3), padding='Same', activation='relu',input_shape=(350,350,3)))
model.add(Conv2D(filters= 64, kernel_size=(3,3), padding='Same', activation='relu'))
# 池化层,池化核大小２x2
model.add(MaxPool2D(pool_size=(2,2)))
# 随机丢弃四分之一的网络连接，防止过拟合
model.add(Dropout(0.25))  
model.add(Conv2D(filters= 128, kernel_size=(2,2), padding='Same', activation='relu'))
model.add(Conv2D(filters= 128, kernel_size=(2,2), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides = (2,2)))
model.add(Conv2D(filters= 256, kernel_size=(2,2), padding='Same', activation='relu'))
model.add(Conv2D(filters= 256, kernel_size=(2,2), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides = (2,2)))
model.add(Conv2D(filters= 512, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(Conv2D(filters= 512, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(Conv2D(filters= 512, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides = (2,2)))
model.add(Conv2D(filters= 512, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(Conv2D(filters= 512, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(Conv2D(filters= 512, kernel_size=(3,3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides = (2,2)))
# 全连接层,展开操作，
model.add(Flatten())
# 添加隐藏层神经元的数量和激活函数
model.add(Dense(256, activation='relu'))    
model.add(Dropout(0.25))
# 输出层
model.add(Dense(num_classes, activation='softmax'))  
model.compile(optimizer=RMSprop(lr = 0.001, decay=0.0), loss = 'categorical_crossentropy',metrics=['accuracy'])

def training(begin,threshold):
    end = begin+threshold
    if end>len(imgsNames):
        end = len(imgsNames)
    print(begin,end)
train_dir = 'Cloth image Dataset/image/train'
train_imgs = os.listdir(train_dir)
#拿取数据
imgDir = 'Cloth image Dataset/image/train/'
images = []
for number in range(begin,end):
    name = imgsNames[number]
    imgName = imgDir+name
    img = Image.open(imgName)
    img = img.resize((350,350))
    img = img.convert('RGB')
    images.append(np.array(img))
X_train, X_test, y_train, y_test = train_test_split(images[:], labels.values[begin:end,1], test_size = 0.2, random_state = 0)
    del(images)
X_train = np.stack(X_train)
X_test = np.stack(X_test)
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
    data_augment = ImageDataGenerator(rotation_range= 10,zoom_range= 0.1,
                                  width_shift_range = 0.1,height_shift_range = 0.1,
                                  horizontal_flip = False, vertical_flip = False)
    learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 3,
                                            verbose = 1, factor=0.5, min_lr = 0.00001)
    history = model.fit_generator(data_augment.flow(X_train, y_train, batch_size=batch_size),
                             epochs= num_epochs, validation_data = (X_test,y_test),
                             verbose =2, steps_per_epoch=X_train.shape[0]//batch_size,callbacks=[learning_rate_reduction])
    fig,ax = plt.subplots(2,1,figsize=(10,10))
    ax[0].plot(history.history['loss'], color='r', label='Training Loss')
    ax[0].plot(history.history['val_loss'], color='g', label='Validation Loss')
    ax[0].legend(loc='best',shadow=True)
    ax[0].grid(True)
    ax[1].plot(history.history['acc'], color='r', label='Training Accuracy')
    ax[1].plot(history.history['val_acc'], color='g', label='Validation Accuracy')
    ax[1].legend(loc='best',shadow=True)
    ax[1].grid(True)


for i in range(len(imgsNames)//1000):
    training(i*1000,1000)

test_dir = 'Cloth image Dataset/image/test/'
test_imgs = os.listdir(test_dir)
len(test_imgs)

#进行测试
def predict(begin,threshold):
    end = begin+threshold
    if end>len(test_imgs):
        end = len(test_imgs)
    print(begin,end)
    X_train = []
    for number in range(begin,end):
        name = test_imgs[number]
        imgName = test_dir+name
        img = Image.open(imgName)
        img = img.resize((350,350))
        img = img.convert('RGB')
        X_train.append(np.array(img))
    X_train = np.stack(X_train)
    y_predict = model.predict(X_train)
    y_true = []
    for arr in y_predict:
        y_true.append(np.argmax(arr))
    return y_true

import time
ans = []
for i in range(len(test_imgs)//100):
    t1 = time.time()
    y_true = predict(i*100,100)
    ans += y_true
    t2 = time.time()
    print(t2-t1)

y = pd.DataFrame(ans,test_imgs)
y.columns = ['Cloth_label']
y.index.name = 'Image'
y.to_csv('ans.csv')