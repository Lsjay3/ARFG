# Jay的开发时间：2022/9/16  15:01
import numpy as np
# import random
import matplotlib.pyplot as plt
import sys
import os


def denoise(t, x):
    # 1、数据预处理
    res = int(np.sqrt(len(x)))
    xr = x[:res * res]
    delay = t[:res * res]

    # 2、一维数组转换为二维矩阵
    x2list = []
    for i in range(res):
        x2list.append(xr[i * res:i * res + res])
    x2array = np.array(x2list)

    # 3、奇异值分解
    U, S, V = np.linalg.svd(x2array)
    S_list = list(S)
    ## 奇异值求和
    S_sum = sum(S)
    ##奇异值序列归一化
    S_normalization_list = [x / S_sum for x in S_list]

    # 4、画图
    X = []
    for i in range(len(S_normalization_list)):
        X.append(i + 1)

    fig1 = plt.figure().add_subplot(111)
    fig1.plot(X, S_normalization_list)
    fig1.set_xticks(X)
    fig1.set_xlabel('Rank', size=15)
    fig1.set_ylabel('Normalize singular values', size=15)
    plt.show()

    # 5、数据重构
    K = 2  ## 保留的奇异值阶数
    for i in range(len(S_list) - K):
        S_list[i + K] = 0.0

    S_new = np.mat(np.diag(S_list))
    reduceNoiseMat = np.array(U * S_new * V)
    reduceNoiseList = []
    for i in range(len(x2array)):
        for j in range(len(x2array)):
            reduceNoiseList.append(reduceNoiseMat[i][j])

    # 6、返回结果
    return (delay, reduceNoiseList)
