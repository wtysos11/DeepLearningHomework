import math

def calc(w,x):
    calc_x = w.dot(x)#绝对值大于5的时候，为了防止溢出与简化效率，应该直接进行赋值
    if calc_x > 6:
        return 1
    elif calc_x < -6:
        return -1
    ans = 1/1+math.exp(-1*calc_x)
    return ans

from sklearn import datasets
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv

iris = datasets.load_iris()
X = iris.data[:100, :]
y = iris.target[:100].reshape((100, -1))


def logit(x):
    return 1. / (1 + np.exp(-x))


m, n = X.shape
alpha = 0.0065 # 步长
w = np.random.random((n, 1)) # 参数矩阵
maxCycles = 30
J = pd.Series(np.arange(maxCycles, dtype=float))

for i in range(maxCycles):
    h = logit(np.dot(X, w)) # 输出估计值h
    J[i] = -(1 / 100.) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) # 记录目标函数的值
    error = h - y #计算wx的梯度，输出值减去真实值
    grad = np.dot(X.T, error) #计算w的梯度
    w -= alpha * grad # 更新参数w，使用负梯度最小化J
print w
J.plot()
plt.show()