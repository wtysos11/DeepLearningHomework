from multiprocessing import Pool

import numpy as np
import pandas as pd
import math

class RandomForestRegressor:
    '''
    Random Forest Regressor
    '''

    def __init__(self,n_estimators = 100,max_depth = 2,n_jobs = -1):
        self._n_estimators = n_estimators
        self._n_jobs = n_jobs
        self._max_depth = max_depth
        self._trees = [] #生成树列表



    # 训练随机森林集合（并行化处理）
    def fit(self,data,label):
        '''
        data和label均为pandas.DataFrame结构
        考虑到数据存放问题，在本函数内处理数据，并并行化地调用训练生成树
        '''
        for i in range(self._n_estimators):
            myTree = DecisionTreeRegressor(self._max_depth)
            myTree.fit(data,label)
            self._trees.append(myTree)
            print(i)


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
class DecisionTreeRegressor:
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

    class DecisionTreeNode:
        '''
        决策树结构设计：
            决策树的根节点保存在root中
            决策树的每个节点有三个指针，指向parent，以及左右孩子
            决策树的每个节点保留自己分类的特征与范围（回归树节点）
            保留了数据的个数
            保留了深度

            需要保留数据的索引列表（行）以访问数据，不保存数据原本

            如果数据是叶节点，还需要保存平均值作为最后查询时的结果

            如果没有特殊情况，切分时左子树永远小于右子树（所以等于在右边，如果有的话）
        '''
        def __init__(self,dataRow,isLeaf,parent,featureCol,depth):
            self.dataRow = dataRow #拿到的自助法抽到的行的索引
            self.featureCol = featureCol #拿到的基于随机森林思想的特征列的索引
            self.featureNum = len(featureCol)

            self.isLeaf = isLeaf #是否为叶节点
            self.parent = parent
            self.lChild = None
            self.rChild = None
            self.splitCol = 0 #进行分隔的特征的列表
            self.splitVal = 0 #进行分隔的特征的值(左子树小于)
            self.splitJudge = math.inf#MSE
            self.depth = depth

            self.ans = 0
            

    def __init__(self,max_depth):
        self._max_depth = max_depth

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
        self.data = data
        self.label = label
        #建立树
        self._min_leaf_sz = int(len(data)/200) #简单回归树，最多允许2层。因此需要树尽量的平衡，不要出现太小的数据集
        self.root = DecisionTreeRegressor.DecisionTreeNode(self._get_data(data),False,None,self._get_features(data.columns),0)
        self._build_tree(self.root)

    #根据一个节点建立一棵树
    def _build_tree(self,treeNode):
        '''
            根据给定的节点进行树的建立
        '''
        #找到最佳的切分特征和最佳的切分点
        isSplit = self._find_bestSplit(treeNode)
        #判断是否结束建立树的过程（MSE差值、深度信息）
        if (treeNode.depth + 1 < self._max_depth or self._max_depth == -1) and isSplit:
            #如果没有结束，则分成左右子树，并继续递归
            treeNode.isLeaf = False
            lChildRow = [x for x in treeNode.dataRow if self.data[treeNode.splitCol].iloc[x] < treeNode.splitVal] #满足左子树要求的，行索引对应的数据全部小于切分值
            rChildRow = [x for x in treeNode.dataRow if self.data[treeNode.splitCol].iloc[x] >= treeNode.splitVal] #满足右子树要求的，行索引对应的数据全部大于等于切分值
            treeNode.lChild = DecisionTreeRegressor.DecisionTreeNode(lChildRow,False,treeNode,treeNode.featureCol,treeNode.depth+1)
            treeNode.rChild = DecisionTreeRegressor.DecisionTreeNode(rChildRow,False,treeNode,treeNode.featureCol,treeNode.depth+1)
            self._build_tree(treeNode.lChild)
            self._build_tree(treeNode.rChild)
        else:
            #清洗切分痕迹，节点变为叶节点
            treeNode.isLeaf = True
            #计算平均值
            labelSum = self.label[0].iloc[treeNode.dataRow].sum()
            num = len(treeNode.dataRow)
            treeNode.ans = labelSum/num

        #如果结束了，就结束吧

    def _compute_mse(self,square_sum,y_sum,number):
        '''
            自己计算MSE，展开后需要平方和，和以及数量
        '''
        return square_sum/number - (y_sum/number)**2

    def _find_bestSplit(self,treeNode):
        '''
            对于给定的决策树节点，找到最佳的切分
            树是回归树，因此衡量切分使用的是MSE
            即切分要使得两边的MSE的加权和最小。
            由于MSE计算上比较复杂，可能速度会比较慢，一个考虑是使用sklearn的MSE进行加速。如果是自己做的话，展开原式子可以发现MSE = 每个点的平方的和 - 每个点的和的平方除以N
                因此计算MSE可以通过维护平方和、和以及数量三个数据来快速计算

            答案写在treeNode.splitCol和treeNode.splitVal中,MSE保存在splitJudge中

            返回一个bool值，这个值判断切分前后MSE的变化决定是否进行切分。
                如果MSE在最佳切分后反而变大了，那么就取消切分操作。
        '''
        splitDiff = 0
        isSplit = True
        #遍历所有的特征
        for col in treeNode.featureCol:
            #拿到所有的行数据。对于DataFrame，使用DataFrame.values[feature][row]进行操作
            aimData = self.data.values[treeNode.dataRow,col]
            #进行排序和统计，取得遍历数据
            sorted_idx = np.argsort(aimData)
            sorted_data = aimData[sorted_idx]
            sorted_label = self.label.values[sorted_idx,0]
            #准备遍历时需要计算MSE而维护的量
            lchild_number = 0
            lchild_sum = 0
            lchild_square_sum = 0
            rchild_number = len(sorted_label)
            rchild_sum = sorted_label.sum()
            rchild_square_sum = (sorted_label**2).sum()
            #节点的数据，作为后面计算MSE差值时要用到的量
            node_number = rchild_number
            node_sum = rchild_sum
            node_square_sum = rchild_square_sum
            #对所有的值进行遍历，保留自身最小的节点数量
            for i in range(0,node_number-self._min_leaf_sz):
                #遍历的时候计算MSE差值
                xi , yi = sorted_data[i],sorted_label[i]

                #修正数据
                rchild_number -= 1
                rchild_sum -= yi
                rchild_square_sum -= yi**2
                lchild_number += 1
                lchild_sum += yi
                lchild_square_sum += yi**2
                if i < self._min_leaf_sz or xi == sorted_data[i+1]:#如果小于最小切分的叶节点数据数量，或是数据相同，则不进行切分
                    continue

                lchild_mse = self._compute_mse(lchild_square_sum,lchild_sum,lchild_number)
                rchild_mse = self._compute_mse(rchild_square_sum,rchild_sum,rchild_number)
                split_mse = (lchild_number*lchild_mse + rchild_number * rchild_mse)/node_number #MSE的加权和
                
                if split_mse < treeNode.splitJudge:
                    treeNode.splitCol = col
                    treeNode.splitVal = xi
                    treeNode.splitJudge = split_mse
                    splitDiff = node_number * (self._compute_mse(node_square_sum,node_sum,node_number) - split_mse)

        if splitDiff < 0:
            isSplit = False

        return isSplit
        

    #接受数据，进行预测
    def predict(self,data):
        '''
        接受了数据，进行预测(思考了一下要不要做随机抽样，似乎不用吧)
        返回一个一维数据向量
        '''
        ans = np.zeros(len(data))
        #对每一行的数据
        for i in range(len(data)):
            row = data.iloc[i]
            ans[i] = self._get_predict(row)

        return ans

    #接受一个行的数据，在决策树上返回预测值
    def _get_predict(self,row):
        node = self.root
        while not node.isLeaf:
            if row[node.splitCol]<node.splitVal:
                node = node.lChild
            else:
                node = node.rChild

        return node.ans

'''
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

myTree = DecisionTreeRegressor(5)
data_num = 1000
myTree.fit(X_train[:data_num],y_train[:data_num])
y_ans = myTree.predict(X_train[data_num:2*data_num])
y_correct = y_train[data_num:2*data_num].values[:,0]

total = len(y_ans)
correct = 0
for i in range(len(y_ans)):
    if y_ans[i] == y_correct[i]:
        correct = correct+1

print(correct/total)
'''