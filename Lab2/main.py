# -*- coding: utf-8 -*-
"""
逻辑斯蒂回归
生成两个类别的数据进行实验

- [x] 1. 无惩罚项梯度下降
- [x] 2. 无惩罚项牛顿法
- [x] 3. 有惩罚项梯度下降
- [x] 4. 有惩罚项牛顿法
- [x] 5. UCI数据验证
"""

import matplotlib.pyplot as plt
import random
import numpy as np


# 数据点起点终点
DATA_HEAD_X = -10
DATA_END_X = 10
DATA_HEAD_Y = -10
DATA_END_Y = 10

# 直线参数
A = -2
B = 4

# 参数维度
WN = 3

# 数据点个数
DATA_NUM = 100

# 惩罚项
LAMBDA_G = 0.5
LAMBDA_N = 0.5

# 最小误差
EPSILON = 1

# 最大迭代次数
LOOP_MAX = 1000

# 学习速度
ALPHA = 0.001

def original_data(data_head_x=DATA_HEAD_X, data_end_x=DATA_END_X, data_head_y=DATA_HEAD_Y, data_end_y=DATA_END_Y, line_a = A, line_b = B, data_num=DATA_NUM):
    """
    original_data 根据某条直线严格的生成具体的两类数据
    
    [description]
        data_head_x (int, optional): Defaults to DATA_HEAD_X. 从此处开始生成数据
        data_end_x (int, optional): Defaults to DATA_END_X. 到此处结束生成数据
        data_head_y (int, optional): Defaults to DATA_HEAD_Y. 从此处开始生成数据
        data_end_y (int, optional): Defaults to DATA_END_Y. 到此处结束生成数据
        line_a (double, optional): Defaults to A. 生成样本依据的直线的一次项参数
        line_b (double, optional): Defaults to B. 生成样本依据的直线的二次项参数
        data_num (int, optional): Defaults to DATA_NUM. 生成的样本点个数
    
    Returns:
        list, list, mat, mat: 生成的数据分别是x的数据点，y的数据点，xy组成的DATA_NUM*2矩阵，类别矩阵。
    """
    list_x = []
    list_y = []
    for i in range(0, data_num):
        list_x.append(random.uniform(data_head_x, data_end_y))
        list_y.append(random.uniform(data_head_y, data_end_y))
    result = []
    for i in range(data_num):
        if(list_x[i] * line_a + line_b >= list_y[i]):
            result.append(1)
        else:
            result.append(0)
    data_mat = np.vstack((np.mat(list_x), np.mat(list_y)))
    result = np.mat(result)
    return list_x, list_y, data_mat.T, result.T
    

def logistic_regression_Gradient_Descent_without_lambda(data_mat, result, w = WN):
    """
    logistic_regression_Gradient_Descent_without_lambda 无正则化的梯度下降实现逻辑斯蒂回归
    
    [description]
    
    Args:
        data_mat (np.mat): 样本点矩阵 每一行为一个样本的数据
        result (np.mat): 列矩阵 每个样本点的类别 只有01
        w (int, optional): Defaults to WN. 参数维度 两项特征则使用三个参数 类推
    
    Returns:
        np.mat, int, np.mat: 分别是参数列矩阵，迭代次数，loss
    """
    re = 10
    count = 0
    size_y = np.size(data_mat, 0)
    y0 = np.ones((size_y, 1))
    X = np.ones((size_y, 1))
    X = np.hstack((X, data_mat))
    y = result
    W = np.zeros((w, 1))

    while abs(re) > EPSILON and count < LOOP_MAX:
        re = 0
        C = X * W
        Y_W = 1.0 / (1 + np.exp(-C))
        y_1 = y.T * np.log2(Y_W)
        y_2 = (y0 - y).T * np.log2(y0 - Y_W)
        l = y_1 + y_2
        W = W - (ALPHA / w) * X.T * (Y_W - y)
        re = l
        count += 1
        # print(W, re, count)
    print('logistic_regression_Gradient_Descent_without_lambda:\n', W, re, count)
    return W, count, re


def logistic_regression_Newton_methon_without_lambda(data_mat, result, w = WN):
    """
    logistic_regression_Newton_methon_without_lambda 无正则化牛顿法实现逻辑斯蒂回归
    
    [description]
    
    Args:
        data_mat (np.mat): 样本点矩阵 每一行为一个样本的数据
        result (np.mat): 列矩阵 每个样本点的类别 只有01
        w (int, optional): Defaults to WN. 参数维度 两项特征则使用三个参数 类推
    
    Returns:
        np.mat, int, np.mat: 分别是参数列矩阵，迭代次数，loss
    """
    re = 10
    count = 0
    size_y = np.size(data_mat, 0)
    y0 = np.ones((size_y, 1))
    X = np.ones((size_y, 1))
    X = np.hstack((X, data_mat))
    y = result
    W = np.zeros((w, 1))

    while abs(re) > EPSILON and count < LOOP_MAX:
        re = 0
        C = X * W
        Y_W = 1.0 / (1 + np.exp(-C))
        y_1 = y.T * np.log2(Y_W)
        y_2 = (y0 - y).T * np.log2(y0 - Y_W)
        # s = 1.0 / w 
        H = X.T * np.mat(np.diag(Y_W.T.tolist()[0])) * np.mat(np.diag((y0 - Y_W).T.tolist()[0])) * X
        S = X.T * (Y_W - y)
        l = y_1 + y_2
        W = W - (H.I * S)
        re = l
        count += 1
        # print(W, re, count)
    print('logistic_regression_Newton_methon_without_lambda:\n', W, re, count)
    return W, count, re
    

def logistic_regression_Gradient_Descent_with_lambda(data_mat, result, w = WN, Lambda=LAMBDA_G):
    """
    logistic_regression_Gradient_Descent_with_lambda 正则化梯度下降实现逻辑斯蒂回归
    
    [description]
    
    Args:
        data_mat (np.mat): 样本点矩阵 每一行为一个样本的数据
        result (np.mat): 列矩阵 每个样本点的类别 只有01
        w (int, optional): Defaults to WN. 参数维度 两项特征则使用三个参数 类推
        Lambda (double, optional): Defaults to LAMBDA_G. Lambda正则化惩罚参数
    
    Returns:
        np.mat, int, np.mat: 分别是参数列矩阵，迭代次数，loss
    """
    re = 10
    count = 0
    size_y = np.size(data_mat, 0)
    y0 = np.ones((size_y, 1))
    X = np.ones((size_y, 1))
    X = np.hstack((X, data_mat))
    y = result
    W = np.zeros((w, 1))

    while abs(re) > EPSILON and count < LOOP_MAX:
        re = 0
        C = X * W
        Y_W = 1.0 / (1 + np.exp(-C))
        y_1 = y.T * np.log2(Y_W)
        y_2 = (y0 - y).T * np.log2(y0 - Y_W)
        lam = Lambda / (2 * w) * np.sum(np.power(W, 2))
        l = y_1 + y_2 + lam
        W = W - (ALPHA / w) * (X.T * (Y_W - y) + Lambda * W)
        re = l
        count += 1
        # print(W, re, count)
    print('logistic_regression_Gradient_Descent_with_lambda:\n', W, re, count)
    return W, count, re
    

def logistic_regression_Newton_methon_with_lambda(data_mat, result, w = WN, Lambda=LAMBDA_N):
    """
    logistic_regression_Newton_methon_with_lambda 正则化牛顿法实现逻辑斯蒂回归
    
    [description]
    
        Args:
        data_mat (np.mat): 样本点矩阵 每一行为一个样本的数据
        result (np.mat): 列矩阵 每个样本点的类别 只有01
        w (int, optional): Defaults to WN. 参数维度 两项特征则使用三个参数 类推
        Lambda (double, optional): Defaults to LAMBDA_N. Lambda正则化惩罚参数
    
    Returns:
        np.mat, int, np.mat: 分别是参数列矩阵，迭代次数，loss
    """
    re = 10
    count = 0
    size_y = np.size(data_mat, 0)
    y0 = np.ones((size_y, 1))
    X = np.ones((size_y, 1))
    X = np.hstack((X, data_mat))
    y = result
    W = np.zeros((w, 1))
    A = np.eye(w)
    A[0][0] = 0
    
    while abs(re) > EPSILON and count < LOOP_MAX:
        re = 0
        C = X * W
        Y_W = 1.0 / (1 + np.exp(-C))
        y_1 = y.T * np.log2(Y_W)
        y_2 = (y0 - y).T * np.log2(y0 - Y_W) 
        H = X.T * np.mat(np.diag(Y_W.T.tolist()[0])) * np.mat(np.diag((y0 - Y_W).T.tolist()[0])) * X + Lambda / w * A
        S = X.T * (Y_W - y) - Lambda / w * W
        l = y_1 + y_2
        W = W - (H.I * S)
        re = l
        count += 1
        # print(W, re, count)
    print('logistic_regression_Newton_methon_with_lambda:\n', W, re, count)
    return W, count, re
    

def draw(ax, name, list_x, list_y, W, data_num = DATA_NUM, line_a = A, line_b = B, data_head_x = DATA_HEAD_X, data_end_x = DATA_END_X):
    
    for i in range(0, data_num):
        if A * list_x[i] + B > list_y[i]:
            ax.scatter(list_x[i], list_y[i], label='1', s=10, color='red', marker='.', alpha=0.7)
        else:
            ax.scatter(list_x[i], list_y[i], label='2', s=10, color='blue', marker='x', alpha=0.7)
    line_x = np.linspace(data_head_x, data_end_x, 300)
    line_y = line_a * line_x + line_b 
    ax.plot(line_x, line_y, label='line', color='black', linewidth=0.8, alpha=1)
    ax.set_title(name)
#     plt.legend()


def main():
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 6))
    plt.xlim(DATA_HEAD_X, DATA_END_X)
    plt.ylim(DATA_HEAD_Y, DATA_END_Y)
    ax1 = axes[0,0]
    ax2 = axes[0,1]
    ax3 = axes[1,0]
    ax4 = axes[1,1]
    
    list_x, list_y, data_mat, result = original_data()
    W1, count1, re1 = logistic_regression_Gradient_Descent_without_lambda(data_mat, result)
    W2, count2, re2 = logistic_regression_Newton_methon_without_lambda(data_mat, result)
    W3, count3, re3 = logistic_regression_Gradient_Descent_with_lambda(data_mat, result)
    W4, count4, re4 = logistic_regression_Newton_methon_with_lambda(data_mat, result)
    draw(ax1, "logistic_regression_Gradient_Descent_without_lambda", list_x, list_y, W1, line_a=-W1.tolist()[1][0]/W1.tolist()[2][0], line_b=-W1.tolist()[0][0]/W1.tolist()[2][0])
    draw(ax2, "logistic_regression_Newton_methon_without_lambda", list_x, list_y, W2, line_a=-W2.tolist()[1][0]/W2.tolist()[2][0], line_b=-W2.tolist()[0][0]/W2.tolist()[2][0])
    draw(ax3, "logistic_regression_Newton_methon_with_lambda", list_x, list_y, W3, line_a=-W3.tolist()[1][0]/W3.tolist()[2][0], line_b=-W3.tolist()[0][0]/W3.tolist()[2][0])
    draw(ax4, "logistic_regression_Newton_methon_with_lambda", list_x, list_y, W4, line_a=-W4.tolist()[1][0]/W4.tolist()[2][0], line_b=-W4.tolist()[0][0]/W4.tolist()[2][0])
    print(-W1.tolist()[1][0]/W1.tolist()[2][0], -W1.tolist()[0][0]/W1.tolist()[2][0], re1, count1)
    print(-W2.tolist()[1][0]/W2.tolist()[2][0], -W2.tolist()[0][0]/W2.tolist()[2][0], re2, count2)
    print(-W3.tolist()[1][0]/W3.tolist()[2][0], -W3.tolist()[0][0]/W3.tolist()[2][0], re3, count3)
    print(-W4.tolist()[1][0]/W4.tolist()[2][0], -W4.tolist()[0][0]/W4.tolist()[2][0], re4, count4)
    plt.show()
    

if __name__ == '__main__':
    main()
