'''
Softmax 回归。计算accuracy。
'''
from answerMultiLayerPerceptron import buildGraph, lr, wd1, wd2, batchsize
import mnist
import numpy as np
import pickle
from autograd.utils import PermIterator
from util import setseed

setseed(0) # 固定随机数种子以提高可复现性

save_path = "model/mymlp.npy"

X=np.concatenate((mnist.val_X,mnist.trn_X))
Y=np.concatenate((mnist.val_Y,mnist.trn_Y))
trn_X = X[0:60000,:]
trn_Y = Y[0:60000]
test_X = X[60000:,:]
test_Y = Y[60000:]

if __name__ == "__main__":
    graph = buildGraph(Y)
    # 训练
    best_train_acc = 0
    dataloader = PermIterator(trn_X.shape[0], batchsize)
    for i in range(1, 30+1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        for perm in dataloader:
            tX = trn_X[perm]
            tY = trn_Y[perm]
            graph[-1].y = tY
            graph.flush()
            pred, loss = graph.forward(tX)[-2:]
            hatys.append(np.argmax(pred, axis=1))
            ys.append(tY)
            graph.backward()
            graph.optimstep(lr, wd1, wd2)
            losss.append(loss)
        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)

    # 测试
    with open(save_path, "rb") as f:
        graph = pickle.load(f)
    graph.eval()
    graph.flush()
    pred = graph.forward(test_X, removelossnode=1)[-1]
    haty = np.argmax(pred, axis=1)
    print("valid acc", np.average(haty==test_Y))