import os

os.chdir('E:\\code\\homework\\deepLearning\\homework2')
from multiprocessing import freeze_support
#pandas用于处理数据
import pandas as pd                                                         
import numpy as np
#用于分析数据                 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import tree
import time
from sklearn.metrics import r2_score


if __name__ == "__main__":
    freeze_support()
    dataSet = pd.read_csv('train1.csv',header=None)
    labelSet = pd.read_csv('label1.csv',header=None)
    
    X_train, X_test, y_train, y_test = train_test_split(dataSet, labelSet, test_size = 0.05, random_state = 0)

    X_train = X_train[:10000]
    y_train = y_train[:10000]
    X_test = X_test[:1000]
    y_test = y_test[:1000]

    
    ssX = StandardScaler()
    X_train = ssX.fit_transform(X_train)
    ssY = StandardScaler()
    y_train = ssY.fit_transform(y_train)
    X_test = ssX.transform(X_test)
    y_test = ssY.transform(y_test)
    
    
    myTree = tree.RandomForestRegressor(n_estimators=10,max_depth = 5,n_jobs = 4)
    begin = time.time()
    myTree.fit(X_train,y_train)
    end = time.time()
    print('my time:',end-begin)

    y_pred = myTree.predict(X_test)
    print(r2_score(y_test,y_pred))
    
    skLearnTree = RandomForestRegressor(n_estimators = 10,max_features = 3,max_depth = 5,random_state = 0,n_jobs=-1)
    begin = time.time()
    skLearnTree.fit(X_train,y_train.ravel())
    end = time.time()
    print('sklearn time:',end-begin)
    y_pred = skLearnTree.predict(X_test)
    print(r2_score(y_test,y_pred))
    '''
    展开
    data = X_train
    label = y_train
    _min_leaf_sz = 10
    myTree = tree.DecisionTreeRegressor(2)
    treeNode = tree.DecisionTreeRegressor.DecisionTreeNode(myTree._get_data(data),False,None,myTree._get_features(data.columns),0)
    
    
    splitDiff = 0
    isSplit = True
    for col in treeNode.featureCol:
        aimData = data.values[treeNode.dataRow,col]
        sorted_idx = np.argsort(aimData)
        sorted_data = aimData[sorted_idx]
        sorted_label = label.values[sorted_idx,0]
        lchild_number = 0
        lchild_sum = 0
        lchild_square_sum = 0
        rchild_number = len(sorted_label)
        rchild_sum = sorted_label.sum()
        rchild_square_sum = (sorted_label**2).sum()
        node_number = rchild_number
        node_sum = rchild_sum
        node_square_sum = rchild_square_sum
        for i in range(0,node_number-_min_leaf_sz):
            xi , yi = sorted_data[i],sorted_label[i]
            rchild_number -= 1
            rchild_sum -= yi
            rchild_square_sum -= yi**2
            lchild_number += 1
            lchild_sum += yi
            lchild_square_sum += yi**2
            if i < _min_leaf_sz or xi == sorted_data[i+1]:
                continue
            lchild_mse = myTree._compute_mse(lchild_square_sum,lchild_sum,lchild_number)
            rchild_mse = myTree._compute_mse(rchild_square_sum,rchild_sum,rchild_number)
            split_mse = (lchild_number*lchild_mse + rchild_number * rchild_mse)/node_number
            if split_mse < treeNode.splitJudge:
                treeNode.splitCol = col
                treeNode.splitVal = xi
                treeNode.splitJudge = split_mse
                splitDiff = node_number * (myTree._compute_mse(node_square_sum,node_sum,node_number) - split_mse)
                
    lChildRow = [x for x in treeNode.dataRow if data[treeNode.splitCol].iloc[x] < treeNode.splitVal]
    rChildRow = [x for x in treeNode.dataRow if data[treeNode.splitCol].iloc[x] >= treeNode.splitVal]
    anotherTreeNode = tree.DecisionTreeRegressor.DecisionTreeNode(lChildRow,False,treeNode,treeNode.featureCol,treeNode.depth+1)
    
    '''