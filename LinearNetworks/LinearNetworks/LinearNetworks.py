import random

import torch
import numpy as np
from torch.utils import data


def synthetic_data(w, b, num_examples):
    """
    生成y=Xw+b+噪声
    """
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    """
    打乱数据集中的样本并以小批量方式获取数据
    :param batch_size: 批量大小
    :param features: 特征矩阵
    :param labels: 标签向量
    :return: 大小为batch_size的小批量。 每个小批量包含一组特征和标签
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    # print(indices)
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """
    :param y_hat: 预测值
    :param y: 真实值
    :return: 均方误差
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """
    小批量随机梯度下降
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    # features:X,样本 labels:y,标签
    features, labels = synthetic_data(true_w, true_b, 1000)
    print(features[0], labels[0])
    batch_size = 10
    # for X, y in data_iter(batch_size, features, labels):
    #     print(X, '\n', y)
    #     break
    # 初始化参数
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            # sum()操作将l里所有元素相加，由向量转为标量
            # 向量求梯度比较麻烦，故转为标量
            # 可以理解为将本批次的数据的所有梯度相加一次计算出下降的最快的
            l.sum().backward()
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        with torch.no_grad():
            # 每轮损失
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
