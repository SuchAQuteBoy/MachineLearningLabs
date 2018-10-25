# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import random
import numpy as np

"""
逻辑斯蒂回归
生成两个类别的数据进行实验

- [ ] 1. 无惩罚项梯度下降
- [ ] 2. 无惩罚项牛顿法
- [ ] 3. 有惩罚项梯度下降
- [ ] 4. 有惩罚项牛顿法
- [ ] 5. UCI数据验证
"""

# 数据点起点终点
DATA_HEAD_X = -10
DATA_END_X = 10
DATA_HEAD_Y = -10
DATA_END_Y = 10

# 直线参数
A = -1
B = 0

# 参数维度
WN = 2

# 数据点个数
DATA_NUM = 100

# 惩罚项
LAMBDA = 0.5

# 最小误差
EPSILON = 1

# 最大迭代次数
LOOP_MAX = 1000

# 学习速度
ALPHA = 0.0013

def original_data(data_head_x=DATA_HEAD_X, data_end_x=DATA_END_X, data_head_y=DATA_HEAD_Y, data_end_y=DATA_END_Y, line_a = A, line_b = B, data_num=DATA_NUM):
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
    return list_x, list_y, data_mat, result
    

def logistic_regression_Gradient_Descent_without_lambda(data_mat, result, w = WN):
    re = 10
    count = 0
    size_y = np.size(data_mat, 1)
    y0 = np.ones((size_y, 1)).T
    X = np.ones((size_y, 1)).T
    X = data_mat
    y = result
    W = np.ones((w, 1)).T

    while abs(re) > EPSILON and count < LOOP_MAX:
        re = 0
        C = W * X
        Y_W = 1.0 / (1 + np.exp(-C))
        y_1 = y * (np.log2(Y_W).T)
        y_2 = (y0 - y) * (np.log2(y0 - Y_W).T)
        # s = - 1.0 / w 
        l = y_1 + y_2
        # L = s * l
        W = W - (ALPHA / w) * (Y_W - y) * (X.T)
        re = l
        count += 1
        print(W, re, count)
    return W, count, re


# def logistic_regression_Newton_methon_without_lambda(data):
    
#     print("Hello World!")
#     return a, b


# def logistic_regression_Gradient_Descent_with_lambda(data, Lambda=LAMBDA):
    
#     print("Hello World!")
#     return a, b



# def logistic_regression_Newton_methon_with_lambda(data, Lambda=LAMBDA):
    
#     print("Hello World!")
#     return a, b

def draw(ax, list_x, list_y, W, data_num = DATA_NUM, line_a = A, line_b = B, data_head_x = DATA_HEAD_X, data_end_x = DATA_END_X):
    
    for i in range(0, data_num):
        if -1 * list_x[i] + 0 > list_y[i]:
            ax.scatter(list_x[i], list_y[i], label='1', s=10, color='red', marker='.', alpha=0.7)
        else:
            ax.scatter(list_x[i], list_y[i], label='2', s=10, color='blue', marker='x', alpha=0.7)
    line_x = np.linspace(data_head_x, data_end_x, 300)
    line_y = line_a * line_x + line_b 
    ax.plot(line_x, line_y, label='line', color='black', linewidth=0.8, alpha=1)
#     plt.legend()


def main():
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(19.2, 10.8))
    plt.xlim(DATA_HEAD_X, DATA_END_X)
    plt.ylim(DATA_HEAD_Y, DATA_END_Y)
    ax1 = axes[0,0]
    
    list_x, list_y, data_mat, result = original_data()
    W, count, re = logistic_regression_Gradient_Descent_without_lambda(data_mat, result)
    draw(ax1, list_x, list_y, W, line_a=W.tolist()[0][1], line_b=W.tolist()[0][0])
    print(W)
    print(count)
    print(re)
    
    
    plt.show()


if __name__ == '__main__':
    main()
