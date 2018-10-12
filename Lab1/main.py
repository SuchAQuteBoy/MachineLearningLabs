#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

"""
机器学习第一次试验
实现功能：
-[x] 1. 生成数据，加入噪声；
-[x] 2. 用高阶多项式函数拟合曲线；
-[x] 3. 用解析解求解两种loss的最优解（无正则项和有正则项）
-[x] 4. 优化方法求解最优解（梯度下降，共轭梯度）；
-[x] 5. 用你得到的实验数据，解释过拟合。
-[x] 6. 用不同数据量，不同超参数，不同的多项式阶数，比较实验效果。
-[x] 7. 语言不限，可以用matlab，python。求解解析解时可以利用现成的矩阵求逆。
    梯度下降，共轭梯度要求自己求梯度，迭代优化自己写。不许用现成的平台，例
    如pytorch，tensorflow的自动微分工具。
"""

# 测试点区间
TEST_START = 1
TEST_END = 2

# 正则项惩罚参数
LAMBDA = 0.0004

# 计算点区间
CALCULATE_START = 0
CALCULATE_END = 1

# 参数维度
PARAM_NUMBER = 9

# 样本点数
SAMPLE_NUMBER = 10

# 测试点数
TEST_POINTS_NUMBERS = 20

# 梯度下降系列参数
LOOP_MAX = 10000
EPSILON = 0.8
ALPHA = 0.2


def random_point(start, end, a):
    """
    样本点生成函数 通过定义生成区间和生成个数 返回样本点矩阵
    :param start: 正数 必须小于end
    :param end: 正数 必须大雨start
    :param a: 样本点个数 必须是正整数
    :return: 返回2*N的样本点矩阵
    """
    array_x = np.linspace(start, end, a)
    # array_y0 = np.sin(2 * np.pi * array_x) + np.random.randn(a)
    array_y0 = np.sin(2 * np.pi * array_x) + np.random.normal(scale=0.5, size=array_x.shape)

    # print(array_x)
    # print(array_y0)

    result = np.vstack((np.mat(array_x), np.mat(array_y0)))
    # print("result:")
    # print(result)
    return result

    # figure = plt.figure()
    # ax = figure.add_subplot(111)
    #
    # ax.set_title("Figure1")
    #
    # plt.xlabel("X")
    # plt.ylabel("Y")
    #
    # plt.legend("x1")
    #
    # ax.scatter(array_x, array_y0, c="r", marker=".")
    #
    # plt.show()


def random_point_test(start, end, a):
    """
    测试点生成函数 通过定义测试区间生成a个数据点进行测试
    :param start: 正数 必须小于end
    :param end: 正数 必须大雨start
    :param a: 生成样本点数量 必须为正整数
    :return: 返回2*N的数据点矩阵
    """
    array_x = np.linspace(start, end, a)
    array_y0 = np.sin(2 * np.pi * array_x) + np.random.randn(a)
    result = np.vstack((np.mat(array_x), np.mat(array_y0)))
    return result


def analytic_solutions_regular_matching(a, result):

    """
    解析解的无正则计算函数 通过现有的公式进行计算 得出未经过调整的参数结果
    :param a: 参数为最终计算的列矩阵维度 必须为大于1的正整数
    :param result: 样本点矩阵 必须是2*N维度
    :return: 返回参数列矩阵
    """
    size = SAMPLE_NUMBER
    x = np.transpose(result[0])
    y = np.transpose(result[1])
    X = np.ones((size, 1))
    for i in range(1, a):
        X = np.hstack((X, np.power(x, i)))
    Xt = np.transpose(X)
    i = np.eye(a)
    W = (Xt * X + LAMBDA * i).I * Xt * y
    return W


def analytic_solutions_irregular_matching(a, result):

    """
    解析解的无正则计算函数 通过现有的公式进行计算 得出未经过调整的参数结果
    :param a: 参数为最终计算的列矩阵维度 必须为大于1的正整数
    :param result: 样本点矩阵 必须是2*N维度
    :return: 返回参数列矩阵
    """
    size = SAMPLE_NUMBER
    x = np.transpose(result[0])
    y = np.transpose(result[1])
    X = np.ones((size, 1))
    for i in range(1, a):
        X = np.hstack((X, np.power(x, i)))
    Xt = np.transpose(X)
    W = (Xt * X).I * Xt * y
    return W


def stochastic_gradient_descent_matching(a, result):
    """
    SDG匹配
    :param a:参数维度
    :param result:输入样本矩阵 必须为2*N
    :return:返回参数列矩阵
    """
    np.random.seed(a)
    size = SAMPLE_NUMBER
    x = np.transpose(result[0])
    y = np.transpose(result[1])
    X = np.ones((size, 1))
    for i in range(1, a):
        X = np.hstack((X, np.power(x, i)))
    re = 10
    theta = np.ones((a, 1))
    count = 0
    while re > EPSILON and count < LOOP_MAX:
        re = 0
        d = 2 * X.T * X * theta - 2 * X.T * y + LAMBDA * (1 / np.sqrt(theta.T * theta)) * theta
        theta = theta - ALPHA * d
        re = np.linalg.norm(y - X * theta)
        count += 1
    print("theta:", theta)
    print("final loss:", re)
    return theta


def conjugate_gradient(a, result):
    size = SAMPLE_NUMBER
    x = np.transpose(result[0])
    y = np.transpose(result[1])
    X = np.ones((size, 1))
    for i in range(1, a):
        X = np.hstack((X, np.power(x, i)))

    x0 = np.ones((a, 1))
    k = np.eye(a)
    A = X.T * X + LAMBDA * k
    b = X.T * y
    count = 0
    r0 = b - np.dot(A, x0)
    p0 = r0
    # test1 = np.dot(r0.T, r0)
    # test2 = p0.T * X * p0
    while count < 1000:
        alpha0 = np.dot(r0.T, r0) / (p0.T * A * p0)
        x1 = x0 + alpha0.tolist()[0][0] * p0
        r1 = r0 - alpha0.tolist()[0][0] * A * p0
        if np.linalg.norm(r1) < 1e-6:
            print("count:", count)
            print(x1)
            return x1
        beta0 = np.dot(r1.T, r1) / np.dot(r0.T, r0)
        p1 = r1 + beta0.tolist()[0][0] * p0
        x0 = x1
        p0 = p1
        r0 = r1
        count += 1
    print("count:", count)
    return x0


def pre_draw(ax, result):
    plt.xlim(0, 2)
    plt.ylim(-2, 2)
    x0 = np.linspace(CALCULATE_START, TEST_END, 2000)
    y1 = np.sin(2 * np.pi * x0)

    plt.xlabel("X")
    plt.ylabel("Y")

    test = random_point_test(TEST_START, TEST_END, TEST_POINTS_NUMBERS)

    ax.scatter(test[0].tolist(), test[1].tolist(), label="test points", c="y", marker=".")
    ax.scatter(result[0].tolist(), result[1].tolist(), label="calculated points", c="r", marker=".")
    ax.plot(x0, y1, label="sin", color="blue")
    plt.legend()


def draw(s, w, ax, label, color):
    """
    画图函数 已经封装了所有的可能结果 通过传入的参数进行绘制并且会在图形中展示
    测试点和计算点
    :param s: 图标题
    :param ax: 画图的画布区域
    :param w: 计算得出的参数矩阵
    :param label: 图例
    :param color: 颜色算子
    :return: 无返回值 直接调用绘图
    """
    wt = np.transpose(w)
    l = list(reversed(wt.tolist()[0]))
    x0 = np.linspace(CALCULATE_START, TEST_END, 2000)
    y0 = np.poly1d(l)
    print(y0)

    ax.set_title(s)

    ax.plot(x0, y0(x0), label=label, color=[color * 0.04, color * 0.08, color * 0.1])
    plt.legend()


def main():

    global PARAM_NUMBER, LAMBDA
    fig = plt.figure(figsize=(19.2, 10.8))
    result = random_point(CALCULATE_START, CALCULATE_END, SAMPLE_NUMBER)

    print("analytic_solutions_irregular_matching")
    ax1 = fig.add_subplot(221)
    pre_draw(ax1, result)
    for i in range(2, 10):
        PARAM_NUMBER = i
        w = analytic_solutions_irregular_matching(PARAM_NUMBER, result)
        draw("analytic_solutions_irregular_matching", w, ax1, "PARAM_NUMBER:" + str(PARAM_NUMBER), i)

    PARAM_NUMBER = 9

    print("analytic_solutions_regular_matching with PARAM_NUMBER:" + str(PARAM_NUMBER))
    ax2 = fig.add_subplot(222)
    pre_draw(ax2, result)
    for i in range(2, 10):
        LAMBDA = i * 0.0004
        w = analytic_solutions_regular_matching(PARAM_NUMBER, result)
        draw("analytic_solutions_regular_matching with PARAM_NUMBER:" + str(PARAM_NUMBER), w, ax2, "lambda:" + str(LAMBDA), i)

    print("stochastic_gradient_descent_matching")
    ax3 = fig.add_subplot(223)
    pre_draw(ax3, result)
    for i in range(2, 10):
        PARAM_NUMBER = i
        w = stochastic_gradient_descent_matching(PARAM_NUMBER, result)
        draw("stochastic_gradient_descent_matching", w, ax3, "PARAM_NUMBER:" + str(PARAM_NUMBER), i)

    PARAM_NUMBER = 9
    ax4 = fig.add_subplot(224)
    pre_draw(ax4, result)
    w = conjugate_gradient(PARAM_NUMBER, result)

    draw("conjugate_gradient", w, ax4, "a", 4)
    plt.show()


if __name__ == "__main__":
    main()
