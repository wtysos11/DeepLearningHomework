from multiprocessing import Pool

import numpy as np
import pandas as pd
import math
from sklearn.tree import DecisionTreeRegressor

class RandomForestRegressor:
    '''
    Random Forest Regressor
    '''

    def __init__(self,n_estimators = 100,max_depth = 5,n_jobs = -1):
        self._n_estimators = n_estimators
        self._n_jobs = n_jobs
        self._max_depth = max_depth
        self._trees = [] #生成树列表

    def _single_tree_fit(self,tree):
        return tree.fit(self.data,self.label)

    # 训练随机森林集合（并行化处理）
    def fit(self,data,label):
        '''
        data和label均为numpy.array结构
        考虑到数据存放问题，在本函数内处理数据，并并行化地调用训练生成树
        '''
        self.data = data
        self.label = label
        for i in range(self._n_estimators):
            self._trees.append(MyDecisionTreeRegressor(self._max_depth))

        if self._n_jobs == 0:
            for tree in self._trees:
                self._single_tree_fit(tree)
        else:
            worker = self._n_jobs
            if worker == -1:
                worker = 8
            pool = Pool(processes = worker)
            self._trees = list(pool.map(self._single_tree_fit,self._trees))


    # 根据得到的生成树集合进行预测并返回结果
    def predict(self,data):
        answer = np.zeros(len(data))
        for tree in self._trees:
            answer += tree.predict(data)
            
        ans = answer/self._n_estimators
        return ans

#决策树节点


lg2 = math.log(2)

#决策树回归器
class MyDecisionTreeRegressor:
    '''
        决策树训练过程：
            自助法随机产生一个数据集合，用来作为本决策树的数据
            产生一个log2N的特征子集

            开始训练，穷举所有的特征和特征的几乎所有的取值，找到最优的取值
                使用MSE作为衡量标准，最优取值要使得切分后的数据的MSE最小。
                产生新的左子树和右子树。
                    递归在左子树和右子树上调用算法，如果还没有停止的话。
                    如果停止了，则没有产生左子树和右子树，节点为叶节点，记录数据的平均值作为日后查询的结果。
            
            停止条件：
                1. 如果切分完毕后数据的MSE还没有原来大(加权和)，或者小于某个阈值，那么取消这次切分
                2. 到达指定层数
                3. 所有的数据都相同（数据一样自然是不用切分了，不过真的会全部一样吗）
                4. 切分后的数据量太小

            预测：
                使用深度优先搜索遍历整棵树，按照节点的信息进行寻找
                    
    '''

    def __init__(self,max_depth=5):
        self._max_depth = max_depth
        self.tree = DecisionTreeRegressor(max_depth = self._max_depth)

    def _get_data(self,data):
        '''
        决策树内产生数据
        使用自助法抽取数据，产生可以含重复数据的列表
        接受一个pandas.DataFrame或者numpy形的数据，返回随机编号列表
        '''
        return np.random.choice(len(data),len(data))

    def _get_features(self,feature):
        '''
        随机森林选取随机特征
        产生一个特征的排列（肯定不会允许重复的吧）
        数量为log2N
        '''
        return np.random.choice(len(feature),int(math.log(len(feature))/lg2),replace = False)

    #接受数据，训练一棵树
    def fit(self,data,label):
        '''
            提供数据和标签，进行训练
        '''
        #存储数据和标签

        #建立树
        self.dataRow = self._get_data(data)
        self.labelCol = self._get_features(range(data.shape[1]))
        self.data = data[:,self.labelCol]
        self.data = self.data[self.dataRow,:]
        self.label = label
        self.tree.fit(self.data,self.label.ravel())
        print('fit over')
        return self

    #接受数据，进行预测
    def predict(self,data):
        return self.tree.predict(data[:,self.labelCol])


