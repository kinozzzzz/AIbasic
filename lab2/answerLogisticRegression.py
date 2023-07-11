import numpy as np

# 超参数
lr = 1e-2 * 1.2 # 学习率
wd = 1e-3 / 3 # l2正则化项系数

def sign(fx):
    return np.int32(fx>=0) - np.int32(fx<0)

def predict(X, weight, bias):
    """
    使用输入的weight和bias预测样本X是否为数字0
    @param X: n*d 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: d
    @param bias: 1
    @return: n wx+b
    """
    linear = X @ weight + bias
    return sign(linear)

def sigmoid(x):
    return 1 / (np.exp(-x) + 1) 

def loss_function(linear, Y):
    new = 1 + np.exp(-(linear * Y))
    loss = np.sum(np.log(new))
    return loss

def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: n*d 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: d
    @param bias: 1
    @param Y: n 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: n 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: 1 由交叉熵损失函数计算得到
        weight: d 更新后的weight参数
        bias: 1 更新后的bias参数
    """
    linear = X @ weight + bias
    Y_predict = predict(X, weight, bias)
    loss  = loss_function(linear, Y)
    new_weight = (1 - 2*wd*lr) * weight - ((sigmoid(-linear * Y) * (-Y)) @ X) * lr
    new_bias =  bias - np.sum(sigmoid(-linear * Y) * (-Y)) * lr
    return Y_predict, loss, new_weight, new_bias
    
