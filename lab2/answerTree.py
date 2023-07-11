import numpy as np
from copy import deepcopy
from typing import List, Callable

EPS = 1e-6
0.0427227
# 超参数，分别为树的最大深度、熵的阈值、信息增益函数
hyperparams = {"depth":6, "purity_bound":0.66, "gainfunc":"negginiDA"}

def entropy(Y: np.ndarray):
    """
    计算熵
    @param Y: n, 标签向量
    @return: 熵
    """
    _, lb_cnt = np.unique(Y,return_counts=True)
    lb_prop = lb_cnt / Y.shape[0]
    return -np.sum(lb_prop * np.log(lb_prop))

def gain(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算信息增益
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @param idx: 第idx个特征
    @return: 信息增益
    """
    feat = X[:, idx]
    ret = entropy(Y)
    feats , featcnt = np.unique(feat, return_counts=True)
    featp = featcnt / feat.shape[0]
    for idx in range(feats.shape[0]):
        ret = ret - featp[idx] * entropy(Y[feat == feats[idx]])
    return ret

def gainratio(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算信息增益比
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @param idx: 第idx个特征
    @return: 信息增益比
    """
    ret = gain(X, Y, idx) / (entropy(X[:, idx]) + EPS)
    return ret

def giniD(Y: np.ndarray):
    """
    计算基尼指数
    @param Y: n, 样本的label
    @return: 基尼指数
    """
    u, cnt = np.unique(Y, return_counts=True)
    p = cnt / Y.shape[0]
    return 1 - np.sum(np.multiply(p, p))

def negginiDA(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算负的基尼指数增益
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @param idx: 第idx个特征
    @return: 负的基尼指数增益
    """
    feat = X[:, idx]
    ufeat, featcnt = np.unique(feat, return_counts=True)
    featp = featcnt / feat.shape[0]
    ret = 0
    for i, u in enumerate(ufeat):
        mask = (feat == u)
        ret -= featp[i] * giniD(Y[mask])
    ret += giniD(Y)  # 调整为正值，便于比较
    return ret

class Node:
    """
    决策树中使用的节点类
    """
    def __init__(self): 
        self.children = {}          # 子节点
        self.featidx: int = None    # 用于划分的特征
        self.label: int = None      # 叶节点的标签

    def isLeaf(self):
        """
        判断是否为叶节点
        @return:
        """
        return len(self.children) == 0

def buildTree(X: np.ndarray, Y: np.ndarray, unused: List[int], depth: int,
              purity_bound: float, gainfunc: Callable, prefixstr=""):
    # purity_bound : 熵的阈值
    node = Node()
    # print(prefixstr, f"label {root.label} numbers {u} count {ucnt}") #可用于debug
    label, lb_cnt = np.unique(Y, return_counts=True)
    node.label = label[np.argmax(lb_cnt)]
    if depth == 0 or entropy(Y) <= purity_bound:
        return node
    gains = [gainfunc(X, Y, i) for i in unused]
    idx = np.argmax(gains)
    node.featidx = unused[idx]
    # print(gains,node.featidx,gains[idx])
    unused = deepcopy(unused)
    unused.pop(idx)
    feats = X[:, node.featidx]
    ufeat = np.unique(feats)
    # 按选择的属性划分样本集，递归构建决策树
    for feat in ufeat:
        tmp = feats == feat
        new_X = X[tmp]
        new_Y = Y[tmp]
        node.children[feat] = buildTree(new_X,new_Y,unused,depth-1,purity_bound,gainfunc,prefixstr)
    return node

def inferTree(root: Node, x: np.ndarray):
    """
    利用建好的决策树预测输入样本为哪个数字
    @param root: 当前推理节点
    @param x: d*1 单个输入样本
    @return: int 输入样本的预测值
    """
    if root.isLeaf():
        return root.label
    child = root.children.get(x[root.featidx], None)
    return root.label if child is None else inferTree(child, x)

