# Jay的开发时间：2022/8/3  10:15
import csv
import math

import Img_compose
from PIL import Image
import pandas as pd
import xlrd
import numpy as np
import xlwt
from xlutils.copy import copy
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from numpy import *
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from scipy import interpolate
from scipy.signal import find_peaks
from pykalman import KalmanFilter
from sklearn.preprocessing import MinMaxScaler
# print(np.random.rand(5, 5))
# plt.matshow(np.random.rand(5, 5), cmap=plt.get_cmap('Greens'), alpha=0.5)  # , alpha=0.3
# plt.show()
# scipy.signal.savgol_filter(x, window_length, polyorder)
# x为要滤波的信号
# window_length即窗口长度
# 取值为奇数且不能超过len(x)。它越大，则平滑效果越明显；越小，则更贴近原始曲线。
# polyorder为多项式拟合的阶数。
# 它越小，则平滑效果越明显；越大，则更贴近原始曲线。
# 平均滤波
def np_move_avg(a, n, mode="same"):
    return (np.convolve(a, np.ones((n,)) / n, mode=mode))


# 滑动方差
def Variance_stream(y, windows):
    v = []
    # y_col = []
    for i in range(len(y)):
        if i + windows >= len(y):
            break
            # gap = len(y) - i - 1
        else:
            gap = windows
        v.append(np.var(y[i: i + gap]))
    return v
    pass

def normalization(data):
    M_m = np.max(data)-np.min(data)
    return (data-np.min(data)) / M_m
def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma
def Segment_gesture(data):
    h, w, channel = data.shape
    var = list()
    for i in range(h):
        for j in range(w):
            # x = data[i][j]
            # print("源数据", data[i][j])
            # 归一化
            x = normalization(data[i][j])
            # print("标准化数据：", x)
            v = Variance_stream(x, int(channel / 30))
            # print(np.var(v[-1]))
            var.append(v)
            # x = list(np.arange(1, len(v) + 1))
            # plt.plot(x, v, lw=4, ls='-', c='k', alpha=0.1)
            # plt.plot()
            # plt.show()
    var = np.array(var)
    varmax = list()
    for i in range(len(var[0])):
        varmax.append(max(var[:, i]))
    x = list(np.arange(1, len(var[0]) + 1))
    height = mean(varmax)
    print(height)
    peaks, _ = find_peaks(varmax, height=height)
    # print(peaks)
    if len(peaks) <= 1:
        return 0, 0
    print(peaks[0], peaks[-1])
    with open('v.csv', 'w') as file:
        writer = csv.writer(file)
    data = pd.DataFrame(varmax)
    data.to_csv('v.csv')
    plt.plot(x, varmax, lw=4, ls='-', c='k', alpha=0.5)
    plt.savefig(r'E:\笨比j\rfid\Impinj R420\img_data\img' + "var" + '.jpg')
    plt.show()

    return peaks[0], peaks[-1]




# 双线性插值实现 扩大 n * n 倍分辨率
def bilinear_interpolation(data, n):
    if len(data.shape) == 3:
        orig_h, orig_w, channels = data.shape
    else:
        orig_h, orig_w = data.shape
        channels = 1
    # orig_h, orig_w = data.shape
    # channels = 1
    dst = np.zeros((orig_h * n, orig_w * n, channels), dtype=np.float64)
    x = linspace(0, orig_w, orig_w, endpoint= False)
    y = linspace(0, orig_h, orig_h, endpoint= False)
    xx = linspace(0, orig_w, orig_w * n, endpoint= False)
    yy = linspace(0, orig_h, orig_h * n, endpoint= False)
    for i in range(channels):
        f = interpolate.interp2d(x, y, data[:, :, i], kind='cubic')
        dst[:, :, i] = f(xx, yy)

    # fig, ax = plt.subplots(nrows=1, ncols=2)
    # ax[0].matshow(data[:, :, 0], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
    #
    # ax[1].matshow(dst[:, :, 0], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
    # plt.show()
    # fig, ax = plt.subplots(nrows=1, ncols=2)
    # ax[0].matshow(data[:, :, 1], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
    #
    # ax[1].matshow(dst[:, :, 1], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
    # plt.show()
    # fig, ax = plt.subplots(nrows=1, ncols=2)
    # ax[0].matshow(data[:, :, 2], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
    #
    # ax[1].matshow(dst[:, :, 2], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
    # plt.show()
    return dst
def bilinear_interpolate(src, dst_size):
    height_src, width_src, channel_src = src.shape  # (h, w, ch)
    height_dst, width_dst = dst_size  # (h, w)

    """中心对齐，投影目标图的横轴和纵轴到原图上"""
    ws_p = np.array([(i + 0.5) / width_dst * width_src - 0.5 for i in range(width_dst)], dtype=np.float32)
    hs_p = np.array([(i + 0.5) / height_dst * height_src - 0.5 for i in range(height_dst)], dtype=np.float32)
    ws_p = np.repeat(ws_p.reshape(1, width_dst), height_dst, axis=0)
    hs_p = np.repeat(hs_p.reshape(height_dst, 1), width_dst, axis=1)

    """找出每个投影点在原图的近邻点坐标"""
    ws_0 = np.clip(np.floor(ws_p), 0, width_src - 2).astype(np.int64)
    hs_0 = np.clip(np.floor(hs_p), 0, height_src - 2).astype(np.int64)
    ws_1 = ws_0 + 1
    hs_1 = hs_0 + 1

    """四个临近点的像素值"""
    f_00 = src[hs_0, ws_0, :].T
    f_01 = src[hs_0, ws_1, :].T
    f_10 = src[hs_1, ws_0, :].T
    f_11 = src[hs_1, ws_1, :].T

    """计算权重"""
    w_00 = ((hs_1 - hs_p) * (ws_1 - ws_p)).T
    w_01 = ((hs_1 - hs_p) * (ws_p - ws_0)).T
    w_10 = ((hs_p - hs_0) * (ws_1 - ws_p)).T
    w_11 = ((hs_p - hs_0) * (ws_p - ws_0)).T

    """计算目标像素值"""
    dst = (f_00 * w_00).T + (f_01 * w_01).T + (f_10 * w_10).T + (f_11 * w_11).T
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].matshow(src[:, :, 0], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3

    ax[1].matshow(dst[:, :, 0], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
    plt.show()
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].matshow(src[:, :, 1], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3

    ax[1].matshow(dst[:, :, 1], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
    plt.show()
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].matshow(src[:, :, 2], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3

    ax[1].matshow(dst[:, :, 2], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
    plt.show()
    return dst

# 欧拉公式计算S
def Get_signal(rssi_static, filter_unwrap_phase_static, rssi_move, filter_unwrap_phase_move):
    Drssi_static = pow(pow(10, (rssi_static / 10 - 3)), 0.5)
    Drssi_move = pow(pow(10, (rssi_move / 10 - 3)), 0.5)
    cos_phase_static = math.cos(filter_unwrap_phase_static)
    sin_phase_static = math.sin(filter_unwrap_phase_static)
    cos_phase_move = math.cos(filter_unwrap_phase_move)
    sin_phase_move = math.sin(filter_unwrap_phase_move)

    S_actual = pow((Drssi_move * cos_phase_move) - (Drssi_static * cos_phase_static), 2) \
               + pow((Drssi_move * sin_phase_move) - (Drssi_static * sin_phase_static), 2)
    return S_actual

# 计算实际反射信号矩阵
def Compute_reflectsignal(rssi_move, phase_move):
    height, width, c1 = rssi_move.shape
    channels = rssi_move.shape[2]
    # channels = c1 if c1 < c2 else c2
    S_actual = np.zeros((height, width, channels), dtype=np.float64)
    for i in range(height):
        for j in range(width):
            rssi_static = mean(rssi_move[i][j][0:10])
            phase_static = mean(phase_move[i][j][0:10])
            for t in range(channels):
                # 坐标（i + 1, j + 1）在 t 时刻的反射信号强度
                # S_actual[i][j][t] = Get_signal(0, phase_static, (rssi_move[i][j][t] - rssi_static) / 100, phase_move[i][j][t])
                # print(rssi_static, phase_static, rssi_move[i][j][t], phase_move[i][j][t])
                S_actual[i][j][t] = Get_signal(0, phase_static, rssi_move[i][j][t] - rssi_static, phase_move[i][j][t])
                # S_actual[i][j][t] = Get_signal(rssi_static, phase_static, rssi_move[i][j][t], phase_move[i][j][t])

    path = 'E:\笨比J\RFID\Impinj R420\实验数据\二十五标签\反射信号.xls'
    # for t in range(channels):
    #     im = plt.matshow(S_actual[:, :, t], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
    #     plt.show()
    # book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    # # book_prepare = copy(book)
    # sheet = book.add_sheet('Reflect_signal', cell_overwrite_ok=True)
    # print(height * width + 1)
    # for i in range(1, height * width + 1):
    #     sheet.write(0, i, str(i))
    # i = 1
    # for h in range(height):
    #     for w in range(width):
    #         for j, data in enumerate(S_actual[h][w]):
    #             sheet.write(j + 1, i, data)
    #         i += 1
    # book.save(path)
    # save_path = r'E:\笨比j\rfid\Impinj R420\img_ref\\'
    # col = int(math.sqrt(channels))
    # row = int(math.sqrt(channels))
    # for t in range(channels):
    #
    #         im = plt.matshow(S_actual[:, :, t], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
    #         plt.title(t)
    #         plt.savefig(r'E:\笨比j\rfid\Impinj R420\img_ref\img' + str(t) + '.jpg')
    #         # plt.colorbar(im)
    #         plt.close()
    # Img_compose.image_compose(save_path, r'E:\笨比j\rfid\Impinj R420\img_ref\all.jpg', row, col)
    print("反射信号计算完成")
    return S_actual

def Compute_likelihood(S_actual, There_signal):
    h, w, channels = S_actual.shape
    # signal某一时刻的所有标签(12 * 9)实际信号强度
    signal = np.zeros((channels, h * w), dtype=np.float64)
    pearson_c = np.zeros((channels, h * w), dtype=np.float64)  # 存每一个时刻的皮尔逊系数分布
    like_hood = np.zeros((channels, h * w), dtype=np.float64)  # 存每一个时刻的似然估计分布
    for t in range(channels):
        num = 0
        for i in range(h):
            for j in range(w):
                # print("反射信号最大:", (max(S_actual[:, :, t])))
                signal[t][num] = S_actual[i][j][t]
                num += 1
    for t in range(channels):      # 某一时刻 共283个时刻
        # print("signal[t]：", signal[t])
        pearson = list()
        for i in range(h):
            for j in range(w):
                # There_signal[i][j] 坐标(i + 1, j + 1)的理论标签阵列
                # signal[t] t时刻所有标签组成的反射信号
                # print("There_signal[i][j]：", There_signal[i][j])

                # print("理论信号最大:", There_signal[i][j].tolist().index(max(There_signal[i][j])))
                # ls = sorted(signal[t].tolist(), reverse = True)
                # print(ls)
                # print(signal[t].tolist().index(ls[0]))
                # print(signal[t].tolist().index(ls[1]))
                # print(signal[t].tolist().index(ls[2]))
                # print("反射信号最大:", signal[t].tolist().index(max(signal[t])))
                pc = pearsonr(There_signal[i][j], signal[t])
                pearson.append(pc[0])
        # print(pearson.index(max(pearson)))
        pearson = torch.tensor(pearson)
        # print(pearson.shape)
        pearson_c[t] = pearson.numpy()
        I_x_y = F.softmax(pearson, dim=0)
        # print(I_x_y)
        like_hood[t] = I_x_y.numpy()
        # print(like_hood[t])
        # print(like_hood[t].tolist().index(max(like_hood[t])))
    print("似然估计量计算完成")
    return like_hood, pearson_c

# 两矩阵相加
def Matrix_add(matrix1, matrix2):
    total_element = [matrix1[i][j] + matrix2[i][j] for i in range(len(matrix1)) for j in range(len(matrix1))]
    new_matrix = [total_element[x:x+len(matrix1)] for x in range(0, len(total_element), len(matrix1))]
    return new_matrix

# 将该时间段的所有矩阵相加
def Recombination(data, start, end):
    l = len(data[0])
    res = np.zeros(l, dtype=np.float64)
    for i in range(start, end):
        # print(data[i])
        res = np.add(data[i], res)
    return res

# 将手势平均分为三个时间段（开始、中间、结束）
def Get_recombination(pearson, start, end, h, w, save_tuple):
    path_feature, index, gesture_name = save_tuple
    save_path = r'E:\笨比j\rfid\Impinj R420\img_feature\\'
    Img_compose.del_file(save_path)
    path_fea = path_feature + "\\" + gesture_name + "." + str(index) + ".jpg"
    print(path_fea)
    gap = int((end - start) / 3)
    pearson_start = torch.tensor(Recombination(pearson, start, start + gap))
    pearson_middle = torch.tensor(Recombination(pearson, start + gap, start + 2 * gap))
    pearson_end = torch.tensor(Recombination(pearson, start + 2 * gap, end))
    # 矩阵叠加之后进行归一化
    likeli_start = Trans_listToImg(F.softmax(pearson_start, dim=0).numpy(), h, w)
    likeli_middle = Trans_listToImg(F.softmax(pearson_middle, dim=0).numpy(), h, w)
    likeli_end = Trans_listToImg(F.softmax(pearson_end, dim=0).numpy(), h, w)

    likeli = np.zeros((h , w , 3), dtype=np.float64)
    likeli[:, :, 0] = likeli_start
    likeli[:, :, 1] = likeli_middle
    likeli[:, :, 2] = likeli_end
    # 对三个时期矩阵进行双线性插值增加分辨率
    feature = bilinear_interpolation(likeli, 3)

    # 将三个矩阵进行垂直拼接
    tmp = np.concatenate([feature[:, :, 0], feature[:, :, 1]], axis = 0)
    res_like = np.concatenate([tmp, feature[:, :, 2]], axis=0)
    im = plt.matshow(feature[:, :, 0], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去 x 轴刻度
    plt.yticks([])  # 去 y 轴刻度
    plt.savefig(r'E:\笨比j\rfid\Impinj R420\img_feature\1.jpg',bbox_inches='tight', pad_inches = -0.1)
    plt.close()
    im = plt.matshow(feature[:, :, 1], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去 x 轴刻度
    plt.yticks([])  # 去 y 轴刻度
    plt.savefig(r'E:\笨比j\rfid\Impinj R420\img_feature\2.jpg',bbox_inches='tight', pad_inches = -0.1)
    plt.close()
    im = plt.matshow(feature[:, :, 2], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去 x 轴刻度
    plt.yticks([])  # 去 y 轴刻度
    plt.savefig(r'E:\笨比j\rfid\Impinj R420\img_feature\3.jpg',bbox_inches='tight', pad_inches = -0.1)
    plt.close()
    im = plt.matshow(res_like, cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去 x 轴刻度
    plt.yticks([])  # 去 y 轴刻度
    plt.colorbar(im)
    plt.close()
    # Img_compose.image_compose(save_path, r'E:\笨比j\rfid\Impinj R420\img_feature\feature.jpg', 3, 1)
    Img_compose.image_compose(save_path, path_fea, 3, 1)
    return res_like

# 将一维列表转化为h行，w列的numpy
def Trans_listToImg(data, h, w):
    res = np.zeros((h, w), dtype=np.float64)
    num = 0
    for i in range(h):
        for j in range(w):
            res[i, j] = data[num]
            num += 1
    return res

# 线性插值
def Linear_interpolation(data, res_time, maxtime):
    x = res_time
    xvals = np.arange(0, maxtime, 40)

    yinterp = np.interp(xvals, x, data)
    return yinterp

def Data_process(data, res_time, max_time, windows):
    height, width, channels = data.shape

    channel = int (max_time / 40) + 1
    # channel = 1
    # 进行一维线性插值（时间上的插值，解决标签与天线的随机通信），间隔40ms插一次值
    res = np.zeros((height, width, channel), dtype=np.float64)

    for i in range(height):
        for j in range(width):
            # 将相位进行解缠绕处理
            if data[0][0][0] > -10:
                res[i][j] = Linear_interpolation(np.unwrap(data[i][j]), res_time[i][j], max_time)
                # print(np.unwrap(data[i][j]))
                # res[i][j] = np.unwrap(data[i][j])
            else:
                # print(len(data[i][j]), len(res_time[i][j]))
                res[i][j] = Linear_interpolation(data[i][j], res_time[i][j], max_time)
                # res[i][j] = data[i][j]
            # 进行滤波操作
            # print(len(res[i][j]))
            res[i][j] = savgol_filter(res[i][j], windows, 2)

    # 将处理后的数据写入表格
    path = r'E:\笨比J\RFID\Impinj R420\实验数据\二十五标签\数据预处理.xls'
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    # book_prepare = copy(book)
    sheet = book.add_sheet('final_data', cell_overwrite_ok=True)
    for i in range(1, height * width + 1):
        sheet.write(0, i, str(i))
    i = 1
    for h in range(height):
        for w in range(width):
            for j, data in enumerate(res[h][w]):
                sheet.write(j + 1, i, data)
            i += 1
    book.save(path)
    print("数据预处理完成")
    return res

# 计算理论信号矩阵
def Computer_theresignal(h, w):
    channels = h * w
    coordi = list()
    for i in range(h):
        for j in range(w):
            coordi.append((i, j))
    there_signal = np.zeros((h, w, channels), dtype=np.float64)
    # print(coordi[0][0], coordi[0][1])
    num = 0
    for i in range(h):
        for j in range(w):
            for m in range(channels):
                there_signal[i][j][m] = 1 / pow(pow(coordi[num][0] - coordi[m][0], 2) +
                               pow(coordi[num][1] - coordi[m][1], 2) + 4, 2)
            num += 1
    # print(num)
    # for i in range(h):
    #     for j in range(w):
    #         there = there_signal[i][j].reshape((5, 5))
    #         im = plt.matshow(there, cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
    #         plt.show()
    print("理论信号计算完成")
    return there_signal

# data[:, :, t]为t时刻所有坐标组成的矩阵
# data[i][j] 为坐标i, j 所有时刻的似然估计
def Trans_likeToImg(like_hood, h, w):
    l = len(like_hood)
    data = np.zeros((h, w, l), dtype=np.float64)
    for t in range(l):
        num = 0
        for i in range(h):
            for j in range(w):
                data[i][j][t] = like_hood[t][num]
                num += 1
    return data
# 此处data为每一时刻的似然估计阵列
def Get_tracecoord(data, k):
    temp = data
    like_k_max = list()
    coord_k_max = list()
    sum_coord = (0, 0)
    for i in range(k):
        like_k_max.append(np.max(temp))
        coord = np.unravel_index(np.argmax(temp), temp.shape)
        sum_coord = (sum_coord[0] + coord[0] * np.max(temp), sum_coord[1] + coord[1] * np.max(temp))
        coord_k_max.append(coord)
        temp[coord[0]][coord[1]] = 0
    return sum_coord, like_k_max

def Pant_img(like_hood, s, h, w, save_tuple):
    save_path = r'E:\笨比j\rfid\Impinj R420\img\\'
    path_feature, index, gesture_name = save_tuple
    path_fea = path_feature + ".jpg"
    l = len(like_hood)
    k = 5
    # 计算 一行与一列最大容纳多少张图片（图片合并）
    col = int(math.sqrt(l))
    row = int(math.sqrt(l))
    data = Trans_likeToImg(like_hood, h, w)
    # 进行双线性插值，增加图片分辨率
    data = bilinear_interpolation(data, 3)
    Img_compose.del_file(save_path)
    coord_trace = list()
    coord_h = list()
    coord_w = list()
    for t in range(l):
        sum_coord, like_k_max = Get_tracecoord(data[:, :, t], k)
        # coord_trace.append(np.unravel_index(np.argmax(data[:, :, t]), data[:, :, t].shape))
        coord_trace.append((int(sum_coord[0] / sum(like_k_max)), int(sum_coord[1] / sum(like_k_max))))
        coord_h.append((int(sum_coord[0] / sum(like_k_max))))
        coord_w.append((int(sum_coord[1] / sum(like_k_max))))
        plt.matshow(data[:, :, t], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
        plt.title(s + t)
        plt.savefig(r'E:\笨比j\rfid\Impinj R420\img\img' + str(s + t) + '.jpg')
        # plt.colorbar(im)
        plt.close()
    P_trace(coord_w, coord_h, h, w, path_fea)
    # Pant_trace(coord_trace, h, w, path_fea)
    Img_compose.image_compose(save_path, r'E:\笨比j\rfid\Impinj R420\img\all.jpg', row, col)
    # img = Image.open(r'E:\笨比j\rfid\Impinj R420\img\all.jpg')
    # plt.imshow(img)
    # plt.show()
    plt.close()
def Pant_trace(coord_trace, h, w, path_fea):
    trace_list = np.zeros((h * 3, w * 3), dtype=np.float64)
    for i in range(len(coord_trace)):
        h, w = coord_trace[i]
        trace_list[h][w] = 1
    plt.matshow(trace_list, cmap=plt.get_cmap('Blues'), alpha=0.5)
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去 x 轴刻度
    plt.yticks([])  # 去 y 轴刻度
    print(path_fea)
    plt.savefig(path_fea, bbox_inches='tight', pad_inches=-0.1)
    # plt.show()

def P_trace(coord_w, coord_h, h, w, path_fea):
    fig = plt.figure()
    ax = fig.gca()
    font1 = {'size': 23}
    font2 = {'size': 17}
    # 设置X、Y、Z坐标轴的数据范围
    # ax.set_xlim([0, 15])
    # # # ax.set_xticks([0.70, 0.75, 0.80, 0.85])
    # ax.set_ylim([0, 15])
    ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
    ax.invert_yaxis()  # 反转Y坐标轴
    # 添加X、Y、Z坐标轴的标注
    # ax.set_xlabel('X(m)', font1, labelpad=28)
    # ax.set_ylabel('Y(m)', font1, labelpad=28)
    # ax.tick_params(labelsize=18, pad=8)

    # matplotlib实现动态绘图
    # 打开交互模式
    plt.ion()
    last_w, last_h = coord_w[0], coord_h[0]
    distance = pow((h * 3) / 2, 2) + pow((w * 3) / 2, 2)
    for iter in range(len(coord_w)):
        if pow(coord_w[iter] - last_w, 2) + pow(coord_w[iter] - last_w, 2) > distance:
            continue
        kalman_predict(coord_w[iter], coord_h[iter], iter, ax)
        last_w, last_h = coord_w[iter], coord_h[iter]
    # 关闭交互模式
    plt.ioff()
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去 x 轴刻度
    plt.yticks([])  # 去 y 轴刻度
    plt.savefig(path_fea, bbox_inches='tight')
    # plt.show()


def kalman_predict(x, y, iter, ax):
    global t, filtered_state_means0, filtered_state_covariances0, lmx, lmy, lpx, lpy
    kf = KalmanFilter(transition_matrices=np.array([[1, 3], [0, 1]]),
                      observation_matrices=np.array([1, 0]),
                      transition_covariance=0.5 * np.eye(2))
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
        # ax.plot([lmx, cmx], [lmy, cmy], 'r', label='measure', linewidth=4.0)
        ax.plot([lpx, cpx], [lpy, cpy], 'b', label='predict', linewidth=15.0)

        # plt.pause(0.01)
        filtered_state_means0, filtered_state_covariances0 = filtered_state_means, filtered_state_covariances
        lpx, lpy = filtered_state_means[0][0], filtered_state_means[0][1]
        lmx, lmy = cmx, cmy