# Jay的开发时间：2022/8/7  21:54
import xlrd
import numpy as np
import xlwt
from xlutils.copy import copy
# data = xlrd.open_workbook('E:\笨比J\RFID\Impinj R420\实验数据\四标签\四标签数据.xlsx')
# book_prepare = copy(data)
# col = ['unwrap_phase_静止','filter_phase_静止','unwrap_phase_手势','filter_phase_手势']
# sheet = data.sheets()[0]
# nrows = sheet.nrows
import sys,os,math,time
# import matplotlib.pyplot as plt
# from numpy import *
# import cv2
# from scipy import interpolate
# # 二维线性插值样例
# def two(data):
#     filename = 'E:\cp3.png'
#
#     img = cv2.imread(filename)
#     # print(img.shape)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # print(gray.shape)
#     gray32 = cv2.resize(gray, (64, 64))
#
#     plt.figure(figsize=(10, 10))
#     plt.imshow(gray32)
#     # print(gray32[0])
#     # plt.savefig('E:\cp31.png')
#
#     x = arange(3)
#     y = arange(4)
#     print(x, y)
#     print(data)
#     f = interpolate.interp2d(x, y, data, kind='linear')
#
#     xx = arange(0, 3, 1 / 3)
#     print(len(xx))
#     yy = arange(0, 4, 1 / 3)
#     zz = f(xx, yy)
#     plt.figure(figsize=(10, 10))
#     print(zz)
#     # plt.imshow(zz)
#     # plt.savefig('E:\cp32.png')
#
#
# import numpy as np
# import matplotlib as mpl
# import pylab as pl
# from scipy import interpolate
#
#
# def z(x, y):
#     return (x + y) * np.exp(-5 * (x ** 2 + y ** 2))

# print(np.var([1, 2, 3, 4, 5]))
# print(np.var([0.85, 0.87, 0.89, 0.91, 0.93]))
# print(np.var([0.98, 0.98, 0.99, 0.98, 0.6]))
# print(np.var([0.18, 0.18, 0.19, 0.18, 0.19]))
# print(np.var([1, 1.05, 0.7, 0.95, 0.98]))

# x, y = np.mgrid[-1:1:15j, -1:1:15j]
# newfunc = interpolate.interp2d(x, y, z(x, y), kind='cubic')
# xnew = np.linspace(-1, 1, 200)
# ynew = np.linspace(-1, 1, 200)
# fnew = newfunc(xnew, ynew)
# pl.subplot(121)
# im1 = pl.imshow(z(x, y), origin='lower', extent=[-1, 1, -1, 1], cmap=mpl.cm.hot)
# pl.colorbar(im1)
# pl.subplot(122)
# im2 = pl.imshow(fnew, origin='lower', extent=[-1, 1, -1, 1], cmap=mpl.cm.hot)
# pl.colorbar(im2)
# pl.show()
# two()
# import numpy as np
# from scipy.interpolate import interp2d
# import matplotlib.pyplot as plt
#
# x = np.linspace(0, 4, 13)
# y = np.array([0, 2, 3, 3.5, 3.75, 3.875, 3.9375, 4])
# X, Y = np.meshgrid(x, y)
# Z = np.sin(np.pi*X/2) * np.exp(Y/2)
#
# x2 = np.linspace(0, 4, 65)
# y2 = np.linspace(0, 4, 65)
# f = interp2d(x, y, Z, kind='cubic')
# Z2 = f(x2, y2)
#
# fig, ax = plt.subplots(nrows=1, ncols=2)
# ax[0].matshow(Z, cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
#
# X2, Y2 = np.meshgrid(x2, y2)
# ax[1].matshow(Z2, cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
#
# plt.show()
#
# import os
#
# if __name__ == '__main__':
#     # Change this to your CSV file base directory'
#     base_directory = 'E:\\笨比J\\RFID\\Impinj R420\\Data\\原始数据\\siwpe_l_to_r'
#
#     for dir_path, dir_name_list, file_name_list in os.walk(base_directory):
#         print(1)
#         for file_name in file_name_list:
#
#             # If this is not a CSV file
#             if not file_name.endswith('.csv'):
#                 # Skip it
#                 continue
#             file_path = os.path.join(dir_path, file_name)
#             with open(file_path, 'r') as ifile:
#                 line_list = ifile.readlines()
#             with open(file_path, 'w') as ofile:
#                 ofile.writelines(line_list[2:])
import matplotlib.pyplot as plt
import numpy as np
from pykalman import KalmanFilter

# 速度的系数使预测的曲线更加贴近于真实曲线
# Q越大，越相信测量值；越小，越相信上一次的估计值
kf = KalmanFilter(transition_matrices=np.array([[1, 3], [0, 1]]),
                  observation_matrices=np.array([1, 0]),
                  transition_covariance=0.1 * np.eye(2))
# transition_matrices是A，     有几个状态就是几维的方阵
# 因为认为是匀速运动，所以B是0
# transition_covariance是Q，状态转移协方差矩阵。    有几个状态就是几维的对角阵 (需要调参)
# observation_matrices是H，观察矩阵。   m*n，m表示能观察到几个状态，n表示状态的个数
# print(np.eye(2))

real_y = [6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 4, 4, 3, 3, 3, 4, 8, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 8, 8, 8, 8, 7, 7, 7, 6, 6, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 12, 12, 12, 10, 10, 10, 10, 10, 10, 10, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
real_x = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 8, 8, 8, 8, 7, 6, 6, 5, 4, 3, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 6, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 12, 12, 12, 12, 12, 12, 11, 11, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
# real_z = [1.1099, 1.0949, 1.0951, 1.095, 1.12079, 1.13055, 1.15717, 1.157966, 1.161549, 1.169428, 1.18917, 1.2265,
#           1.2259, 1.2523, 1.2479, 1.2391, 1.2186, 1.1986, 1.17840, 1.1578, 1.1376, 1.11746]


# real_x = [0.78, 0.7849, 0.7833, 0.7808, 0.77886, 0.769, 0.7696, 0.767, 0.76652, 0.76367, 0.764, 0.7483, 0.75, 0.7646, 0.7646, 0.7588, 0.7588, 0.7588, 0.7588, 0.7588, 0.7588, 0.75778, 0.75778, 0.75549, 0.7528, 0.7528, 0.7533, 0.7555, 0.755, 0.756]
# real_y = [-0.04, -0.0799, -0.115, -0.1768, -0.20688, -0.2331, -0.2596, -0.2788, -0.30189, -0.310, -0.3171, -0.3312, -0.339, -0.3644, -0.3644, -0.38798, -0.38798, -0.38798, -0.38798, -0.38798, -0.38798, -0.3923, -0.3923, -0.392, -0.39, -0.3908, -0.3924, -0.3955, -0.3959, -0.3965]
# real_z = [1.13, 1.132, 1.1326, 1.13344, 1.13369, 1.13678, 1.1367, 1.137, 1.13747, 1.1384, 1.138, 1.1434, 1.14192, 1.1347, 1.1347, 1.1391, 1.1391, 1.1391, 1.1391, 1.1391, 1.1391, 1.1392, 1.1392, 1.1397, 1.1405, 1.1405, 1.1402, 1.1393, 1.1393, 1.1391]

def kalman_predict(x, y, iter):
    global t, filtered_state_means0, filtered_state_covariances0, lmx, lmy, lpx, lpy

    if iter == 0:
        # 上一时刻的状态X(k-1)
        filtered_state_means0 = np.array([[x, y], [0.0, 0.0]])
        # 矩阵P初始值
        filtered_state_covariances0 = np.eye(2)
        lmx, lmy = x, y
        lpx, lpy = x, y
    else:
        dx = (x - lmx, y - lmy)
        # 跟矩阵H对应着，Zk
        next_measurement = np.array([x, y])
        # next_measurement = np.array([x, y, z])
        cmx, cmy = x, y
        filtered_state_means, filtered_state_covariances = (
            kf.filter_update(filtered_state_means0, filtered_state_covariances0, next_measurement,
                             transition_offset=np.array([0, 0])))
        cpx, cpy = filtered_state_means[0][0], filtered_state_means[0][1]
        # 绘制真实轨迹和卡尔曼预测轨迹，红色是测量值，绿色是预测值
        ax.plot([lmx, cmx], [lmy, cmy], 'r', label='measure', linewidth=4.0)
        ax.plot([lpx, cpx], [lpy, cpy], 'g', label='predict', linewidth=3.0)
        # plt.pause(0.01)
        filtered_state_means0, filtered_state_covariances0 = filtered_state_means, filtered_state_covariances
        lpx, lpy = filtered_state_means[0][0], filtered_state_means[0][1]
        lmx, lmy = cmx, cmy


# if __name__ == '__main__':
#     fig = plt.figure()
#     ax = fig.gca()
#     font1 = {'size': 23}
#     font2 = {'size': 17}
#     # 设置X、Y、Z坐标轴的数据范围
#     ax.set_xlim([0, 15])
#     # ax.set_xticks([0.70, 0.75, 0.80, 0.85])
#     ax.set_ylim([0, 15])
#     ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
#     ax.invert_yaxis()  # 反转Y坐标轴
#     # 添加X、Y、Z坐标轴的标注
#     ax.set_xlabel('X(m)', font1, labelpad=28)
#     ax.set_ylabel('Y(m)', font1, labelpad=28)
#     ax.tick_params(labelsize=18, pad=8)

    # matplotlib实现动态绘图
    # # 打开交互模式
    # plt.ion()
    # for iter in range(len(real_x)):
    #     kalman_predict(real_x[iter], real_y[iter], iter)
    # # 关闭交互模式
    # plt.ioff()
    # plt.show()
