# -*- coding: utf-8 -*-

"""
## 机器学习第三次试验

实现功能：

实现一个k-means算法和混合高斯模型，并且用EM算法估计模型中的参数

  - [x] 1. 用高斯分布产生k个高斯分布的数据（不同均值和方差）
  - [x] 2. 用k-means聚类，测试效果
  - [ ] 3. 用混合高斯模型和你实现的EM算法估计参数，看看每次迭代后似然值变化情况，考察EM算法是否可以获得正确的结果（与你设定的结果比较）
  - [ ] 4. 可以UCI上找一个简单问题数据，用你实现的GMM进行聚类。
"""

import numpy as np
import random
import copy
import matplotlib.pyplot as plt


# K-means 类
GROUP = 3

# 最大迭代
MAX_LOOP = 100

# 维度
WN = 2

# 中心变化
RE = 0.5

# 每一类数据点个数
DATA_NUM = 30

# 核心样本点生成区间
DATA_BEGIN_X = -10
DATA_END_X = 10
DATA_BEGIN_Y = -10
DATA_END_Y = 10

def create_data(group = GROUP, data_begin_x = DATA_BEGIN_X, data_end_x = DATA_END_X, data_begin_y = DATA_BEGIN_Y, data_end_y = DATA_END_Y, data_num = DATA_NUM):
    """
    create_data [summary]
    
    [description]
    
    Keyword Arguments:
        group {[type]} -- [description] (default: {GROUP})
        data_begin_x {[type]} -- [description] (default: {DATA_BEGIN_X})
        data_end_x {[type]} -- [description] (default: {DATA_END_X})
        data_begin_y {[type]} -- [description] (default: {DATA_BEGIN_Y})
        data_end_y {[type]} -- [description] (default: {DATA_END_Y})
        data_num {[type]} -- [description] (default: {DATA_NUM})
    
    Returns:
        [type] -- [description]
    """
    list_x = []
    list_y = []
    x = []
    y = []
    x.extend(np.linspace(data_begin_x, data_end_x, group))
    for i in range(group):
        y.append(random.uniform(data_begin_y, data_end_y))
    for i in range(group):
        list_x.extend(np.random.normal(loc=x[i], scale=random.uniform(1, 3), size=data_num))
        list_y.extend(np.random.normal(loc=y[i], scale=random.uniform(1, 3), size=data_num))
    data_mat = np.vstack((np.mat(list_x), np.mat(list_y)))
    center = np.vstack((np.mat(x), np.mat(y)))
    return list_x, list_y, center.T, data_mat.T

def mat_min(mat):
    """
    mat_min [summary]
    
    [description]
    
    Arguments:
        mat {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    index = 0
    min = 100000
    size = np.size(mat, 0)
    for i in range(size):
        if mat[i] <= min:
            min = mat[i]
            index = i
    return index


def Kmeans(data_mat, k=GROUP, w=WN, max_loop=MAX_LOOP, r = RE):
    """
    Kmeans [summary]
    
    [description]
    
    Arguments:
        data_mat {[type]} -- [description]
    
    Keyword Arguments:
        k {[type]} -- [description] (default: {GROUP})
        w {[type]} -- [description] (default: {WN})
        max_loop {[type]} -- [description] (default: {MAX_LOOP})
        r {[type]} -- [description] (default: {RE})
    
    Returns:
        [type] -- [description]
    """
    center = np.zeros((k, w))
    for i in range(k):
        for j in range(w):
            center[i][j] = np.random.normal(0, 5)
    new_center = copy.copy(center)
    size = np.size(data_mat, 0)
    c = np.zeros((size, 1))
    count = 0
    re = 10
    while re > 0 and count < max_loop:
        distence = np.zeros((k, 1))
        for i in range(size):
            for j in range(k):
                distence[j] = np.sum(np.power((data_mat[i] - new_center[j]), 2))
            c[i] = mat_min(distence)
        
        for i in range(k):
            t = np.zeros((1, w))
            d = 0
            for j in range(size):
                if c[j] == i:
                    t += data_mat[j]
                    d += 1.
            if d == 0:
                continue
            new_center[i] = t / d
        re = np.sum(abs(center - new_center))
        center = copy.copy(new_center)
        # print(center)
        count += 1
    print(count, center, re)
    return center, c


# def EM():
    

    
def main():
    list_x, list_y, center, data_mat = create_data()
    ce, c = Kmeans(data_mat)
    plt.figure(figsize=(8, 6))
    for i in range(GROUP):
        for j in range(DATA_NUM):
            plt.scatter(list_x[i*DATA_NUM + j], list_y[i*DATA_NUM + j], color='r' if i==0 else 'g' if i ==1 else 'b',marker='.', alpha=0.7)
    for i in range(GROUP):
        plt.scatter(ce[i][0], ce[i][1], marker='x', c='black')
    plt.show()


if __name__ == '__main__':
    main()
