from numpy.random import rand
import mnist
from answerTree import *
import numpy as np

# 超参数
num_tree = 15     # 树的数量
ratio_data = 1   # 采样的数据比例
ratio_feat = 0.5 # 采样的特征比例
hyperparams = {"depth":6, "purity_bound":0, "gainfunc":"negginiDA"}
hyperparams["gainfunc"] = eval(hyperparams["gainfunc"])

def buildtrees(X: np.ndarray, Y:np.ndarray):
    """
    构建随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @return: List of DecisionTrees, 随机森林
    """
    # 提示：整体流程包括样本扰动、属性扰动和预测输出
    trees = [] 
    #print(X.shape,Y.shape)
    data_size = np.int32(Y.shape[0] * ratio_data)
    feat = range(0,X.shape[1])
    feat_size = np.int32(X.shape[1] * ratio_feat)
    for _ in range(num_tree):
        random_idx = np.random.choice(range(0,X.shape[0]),size=data_size)
        new_X = X[random_idx]
        new_Y = Y[random_idx]
        random_feat = np.random.choice(feat,size=feat_size,replace=False)
        random_feat = list(map(int,random_feat))
        trees.append(buildTree(new_X,new_Y,random_feat,**hyperparams))
    return trees

def infertrees(trees, X):
    """
    随机森林预测
    @param trees: 随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @return: n, 预测的label
    """
    pred = [inferTree(tree, X)  for tree in trees]
    pred = list(filter(lambda x: not np.isnan(x), pred))
    upred, ucnt = np.unique(pred, return_counts=True)
    return upred[np.argmax(ucnt)]
