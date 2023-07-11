import mnist
from copy import deepcopy
from typing import List
from autograd.BaseGraph import Graph
from autograd.utils import buildgraph
from autograd.BaseNode import *

# 超参数
lr = 1e-3   # 学习率
wd1 = 1e-5  # L1正则化
wd2 = 1  # L2正则化
batchsize = 128

def buildGraph(Y):
    """
    建图
    @param Y: n 样本的label
    @return: Graph类的实例, 建好的图
    """
    graph = [Linear(784,300),relu(),BatchNorm(300),Linear(300,10),LogSoftmax(),NLLLoss(Y)]
    return Graph(graph)