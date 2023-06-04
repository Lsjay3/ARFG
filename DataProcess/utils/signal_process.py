# Jay的开发时间：2022/8/3  10:15
import csv
import math
import os

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import ruptures as rpt
import torch
import torch.nn.functional as F
import xlwt
from config.config import Config, Path
from numpy import *
from pykalman import KalmanFilter
from scipy import interpolate
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from utils.img_process import image_compose

filtered_state_means0 = None
filtered_state_covariances0 = None
lmx = None
lmy = None
lpx = None
lpy = None

"""
    # scipy.signal.savgol_filter(x, window_length, polyorder)
    # x为要滤波的信号
    # window_length即窗口长度
    # 取值为奇数且不能超过len(x)。它越大，则平滑效果越明显；越小，则更贴近原始曲线。
    # polyorder为多项式拟合的阶数。
    # 它越小，则平滑效果越明显；越大，则更贴近原始曲线。
"""


def variance_stream(y, windows):
    """
    计算滑动方差
    :param y: 归一化后的相位
    :param windows: 滑动窗口大小
    :return: type: list
    """
    v = []
    for i in range(len(y)):
        if i + windows >= len(y):
            break
        else:
            gap = windows
        v.append(np.var(y[i: i + gap]))
    return v


def normalization(data: numpy.ndarray):
    """
    归一化
    :param data: 指定坐标的相位值
    :return: 归一化后结果 type: numpy
    """
    M_m = np.max(data) - np.min(data)
    return (data - np.min(data)) / M_m


def standardization(data: numpy.ndarray):
    """
    标准化
    :param data: 指定坐标的相位值
    :return: 标准化后结果
    """
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma


def segment_gesture(phase_move: numpy.ndarray):
    """
    基于贝叶斯变点检测的手势分割方法
    :param phase_move: 预处理完成的phase值
    :return:
    """
    num_labels = phase_move.shape[0] * phase_move.shape[1]
    # 重塑数据为 (num_labels, t) 的数组
    reshaped_data = phase_move.reshape(num_labels, -1)

    start_points = []
    end_points = []

    for label_data in reshaped_data:
        # 选择模型
        algo = rpt.Pelt(model="rbf").fit(label_data)
        # 选择最佳变点
        result = algo.predict(pen=10)

        # 添加第一个变点到 start_points
        if len(result) >= 1:
            start_points.append(result[0])
        # 添加倒数第二个变点到 end_points
        if len(result) >= 2:
            end_points.append(result[-2])

    # 计算平均值
    start = int(np.mean(start_points)) if start_points else 0
    end = int(np.mean(end_points)) if end_points else 1

    return start, end


def bilinear_interpolation(feature_map: numpy.ndarray, bio_resolution: int):
    """
    双线性插值实现 扩大分辨率
    :param feature_map: RFID手指踪迹特征矩阵 (5, 5, 165)
    :param bio_resolution: 单维度插值的分辨率
    :return: 扩大分辨率后的特征矩阵(15, 15, 165)
    """
    if len(feature_map.shape) == 3:
        orig_h, orig_w, channels = feature_map.shape
    else:
        orig_h, orig_w = feature_map.shape
        channels = 1
    dst = np.zeros(
        (orig_h *
         bio_resolution,
         orig_w *
         bio_resolution,
         channels),
        dtype=np.float64)
    x = linspace(0, orig_w, orig_w, endpoint=False)
    y = linspace(0, orig_h, orig_h, endpoint=False)
    xx = linspace(0, orig_w, orig_w * bio_resolution, endpoint=False)
    yy = linspace(0, orig_h, orig_h * bio_resolution, endpoint=False)
    for i in range(channels):
        f = interpolate.interp2d(x, y, feature_map[:, :, i], kind='cubic')
        dst[:, :, i] = f(xx, yy)
    return dst


def compute_signal(rssi_static, filter_unwrap_phase_static,
                   rssi_move, filter_unwrap_phase_move):
    """
    欧拉公式计算坐标（i + 1, j + 1）在 t 时刻的反射信号强度
    :param rssi_static: 静止时的RSSI值，置0
    :param filter_unwrap_phase_static: 静止时且解包后的相位，取rssi_move前十个时刻phase的平均
    :param rssi_move: 手势状态的RSSI值
    :param filter_unwrap_phase_move: 手势时且解包后的相位
    :return:
    S_actual: 坐标（i + 1, j + 1）在 c 时刻的反射信号强度 type = numpy.float64
    """
    rssi_static_value = pow(pow(10, (rssi_static / 10 - 3)), 0.5)
    rssi_move_value = pow(pow(10, (rssi_move / 10 - 3)), 0.5)
    cos_phase_static = math.cos(filter_unwrap_phase_static)
    sin_phase_static = math.sin(filter_unwrap_phase_static)
    cos_phase_move = math.cos(filter_unwrap_phase_move)
    sin_phase_move = math.sin(filter_unwrap_phase_move)

    S_actual = pow((rssi_move_value * cos_phase_move) - (rssi_static_value * cos_phase_static), 2) \
        + pow((rssi_move_value * sin_phase_move) -
              (rssi_static_value * sin_phase_static), 2)
    return S_actual


def compute_reflect_signal(rssi_move: numpy.ndarray,
                           phase_move: numpy.ndarray):
    """
    计算实际反射信号矩阵
    :param rssi_move: 预处理结束的RSSI值
    :param phase_move: 预处理结束的Phase值
    :return:
    S_actual: 一个height * width * channels 的反射信号矩阵，type = np.float64
    Notes: channels代表一个手势数据有多少个采样点（也可称为一个手势持续多少时刻）
    """
    height, width, c1 = rssi_move.shape
    channels = rssi_move.shape[2]
    S_actual = np.zeros((height, width, channels), dtype=np.float64)
    for i in range(height):
        for j in range(width):
            rssi_static = mean(rssi_move[i][j][0:10])
            phase_static = mean(phase_move[i][j][0:10])
            for c in range(channels):
                # 坐标（i + 1, j + 1）在 c 时刻的反射信号强度
                S_actual[i][j][c] = compute_signal(rssi_static=0,
                                                   filter_unwrap_phase_static=phase_static,
                                                   rssi_move=rssi_move[i][j][c] -
                                                   rssi_static,
                                                   filter_unwrap_phase_move=phase_move[i][j][c])
    return S_actual


def compute_likelihood(actual_signal_array: numpy.ndarray,
                       theory_signal_array: numpy.ndarray):
    """
    根据实际信号矩阵和理论信号矩阵计算最大似然相似矩阵，用于估计每个时刻手指在标签阵列的坐标
    :param actual_signal_array: 实际信号矩阵 (height * width * channels)
    :param theory_signal_array: 理论信号矩阵 (height * width * (height * width))
    :return:
    like_hood: 最大似然估计矩阵
    pearson_c: 皮尔逊相关系数矩阵
    """
    h, w, channels = actual_signal_array.shape
    # signal某一时刻的所有标签实际信号强度
    signal = np.zeros((channels, h * w), dtype=np.float64)
    # 每一个时刻的皮尔逊系数分布
    pearson_c = np.zeros((channels, h * w), dtype=np.float64)
    # 每一个时刻的似然估计分布
    like_hood = np.zeros((channels, h * w), dtype=np.float64)
    for c in range(channels):
        num = 0
        for i in range(h):
            for j in range(w):
                # print("反射信号最大:", (max(S_actual[:, :, t])))
                signal[c][num] = actual_signal_array[i][j][c]
                num += 1
    for c in range(channels):      # 某一时刻 共283个时刻
        pearson = list()
        for i in range(h):
            for j in range(w):
                pc = pearsonr(theory_signal_array[i][j], signal[c])
                pearson.append(pc[0])
        pearson = torch.tensor(pearson)
        pearson_c[c] = pearson.numpy()
        I_x_y = F.softmax(pearson, dim=0)
        like_hood[c] = I_x_y.numpy()

    return like_hood, pearson_c


def linear_interpolation(
        data: numpy.ndarray, res_time: numpy.ndarray, maxtime: float):
    """
    线性插值 插值时间间隔为Config.LINEAR_INTERPOLATION_TIME
    :param data: 需要插值的原始信号（RSSI or Phase）
    :param res_time: 该原始信号值的对应时间戳
    :param maxtime: 该状态下读取信号的最大时间戳
    :return:
    y_interp: 当前坐标标签的插值后value
    """
    x = res_time
    x_values = np.arange(0, maxtime, Config.LINEAR_INTERPOLATION_TIME)

    y_interp = np.interp(x_values, x, data)
    return y_interp


def signal_preprocess(data: numpy.ndarray,
                      res_time: numpy.ndarray, max_time: float):
    """
    数据预处理：一维线性插值（时间上）、相位解缠绕、平滑滤波
    :param data: 原始信号（RSSI、Phase）值组成的numpy
    :param res_time: 该原始信号值的对应时间戳
    :param max_time: 该状态下读取信号的最大时间戳
    :return:
    res: 预处理之后的数据，其阅读器响应时间均等
    """
    height, width, channels = data.shape

    channel = int(max_time / Config.LINEAR_INTERPOLATION_TIME) + 1
    # 进行一维线性插值（时间上的插值，解决标签与天线的随机通信），间隔40ms插一次值
    res = np.zeros((height, width, channel), dtype=np.float64)

    for i in range(height):
        for j in range(width):
            # 将相位进行解缠绕处理
            if data[0][0][0] > -10:
                res[i][j] = linear_interpolation(
                    np.unwrap(data[i][j]), res_time[i][j], max_time)
            else:
                res[i][j] = linear_interpolation(
                    data[i][j], res_time[i][j], max_time)
            # 进行滤波操作
            res[i][j] = savgol_filter(res[i][j], Config.FILTER_WINDOWS, 2)

    # 将处理后的数据写入表格
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('final_data', cell_overwrite_ok=True)
    for i in range(1, height * width + 1):
        sheet.write(0, i, str(i))
    i = 1
    for h in range(height):
        for w in range(width):
            for j, data in enumerate(res[h][w]):
                sheet.write(j + 1, i, data)
            i += 1
    book.save(Path.PATH_PREPROCESSING)

    return res


def computer_theory_signal():
    """
    计算理论信号矩阵
    :return:
    there_signal: 理论信号矩阵 type: width * height * channels
    Notes:
        channels: 每一个坐标位置都需要有一个相应的理论信号矩阵
    """
    channels = Config.HEIGHT * Config.WIDTH
    coord = list()
    for i in range(Config.HEIGHT):
        for j in range(Config.WIDTH):
            coord.append((i, j))
    there_signal = np.zeros(
        (Config.HEIGHT,
         Config.WIDTH,
         channels),
        dtype=np.float64)
    num = 0
    for i in range(Config.HEIGHT):
        for j in range(Config.WIDTH):
            for m in range(channels):
                there_signal[i][j][m] = 1 / pow(pow(coord[num][0] - coord[m][0], 2) +
                                                pow(coord[num][1] - coord[m][1], 2) + 4, 2)
            num += 1

    return there_signal


def trans_likelihood_to_feature(like_hood: numpy.ndarray, h: int, w: int):
    """
    将似然估计转化成一张张可视化特征矩阵，便于绘图
    :param like_hood: 似然估计矩阵，like_hood.shape ==（channels * (h * w)）
    :param h: 标签阵列的高
    :param w: 标签阵列的宽
    :return: feature_map: 最后的可视化特征 feature_map.shape == （h * w * channels）
            feature_map[:, :, t]为t时刻所有坐标组成的矩阵
            feature_map[i][j] 为坐标i, j 所有时刻的似然估计
    """
    channels = len(like_hood)
    feature_map = np.zeros((h, w, channels), dtype=np.float64)
    for channel in range(channels):
        num = 0
        for i in range(h):
            for j in range(w):
                feature_map[i][j][channel] = like_hood[channel][num]
                num += 1
    return feature_map


def get_trace_coord(feature_map: numpy.ndarray, k: int):
    """
    手指踪迹预测
    从每个时刻的特征阵列中取前k大值用于预测该时刻手指所在位置
    :param feature_map: 某一时刻的手指踪迹特征阵列
    :param k: 取前k大似然估计值
    :return:
    sum_coord: 该时刻手指最有可能的坐标
    like_k_max: 前k大的特征值
    """
    temp = feature_map
    like_k_max = list()
    coord_k_max = list()
    sum_coord = (0, 0)
    for i in range(k):
        like_k_max.append(np.max(temp))
        # 找到 temp 中的最大值对应的位置坐标
        coord = np.unravel_index(np.argmax(temp), temp.shape)
        sum_coord = (
            sum_coord[0] +
            coord[0] *
            np.max(temp),
            sum_coord[1] +
            coord[1] *
            np.max(temp))
        coord_k_max.append(coord)
        temp[coord[0]][coord[1]] = 0
    return sum_coord, like_k_max


def pant_img(like_hood: numpy.ndarray, start: int, save_tuple: tuple):
    """
    将手势间段的最大似然估计矩阵绘制成可视化图像
    :param like_hood: 手势间段的最大似然估计矩阵
    :param start: 手势开始时间
    :param save_tuple: （可视化图片存放位置, 图片编号, 手指踪迹名）
    :return: None
    """
    path_feature, index, gesture_name = save_tuple
    path_fea = path_feature + ".jpg"
    length = len(like_hood)
    k = 5
    # 计算 一行与一列最大容纳多少张图片（图片合并）
    col = int(math.sqrt(length))
    row = int(math.sqrt(length))
    feature_map = trans_likelihood_to_feature(
        like_hood=like_hood, h=Config.HEIGHT, w=Config.WIDTH)

    # 进行双线性插值，增加图片分辨率
    feature_map = bilinear_interpolation(
        feature_map=feature_map, bio_resolution=Config.BIO_RESOLUTION)
    del_file(path_data=Path.PATH_IMG_FEATURE)
    coord_trace = list()
    coord_h = list()
    coord_w = list()

    for channel in range(length):
        # 预测手指坐标
        sum_coord, like_k_max = get_trace_coord(
            feature_map=feature_map[:, :, channel], k=k)
        coord_trace.append(
            (int(sum_coord[0] / sum(like_k_max)), int(sum_coord[1] / sum(like_k_max))))
        coord_h.append((int(sum_coord[0] / sum(like_k_max))))
        coord_w.append((int(sum_coord[1] / sum(like_k_max))))
        plt.matshow(feature_map[:, :, channel], cmap=plt.get_cmap(
            'Blues'), alpha=0.5)
        plt.title(start + channel)
        plt.savefig(Path.PATH_IMG_FEATURE + 'img' +
                    str(start + channel) + '.jpg')
        plt.close()
    pant_trace(
        coord_w=coord_w,
        coord_h=coord_h,
        h=Config.HEIGHT,
        w=Config.WIDTH,
        path_fea=path_fea)
    image_compose(
        Path.PATH_IMG_FEATURE,
        Path.PATH_IMG_FEATURE + 'all.jpg',
        row,
        col)
    plt.close()


def pant_trace(coord_w, coord_h, h, w, path_fea):
    """
    根据坐标绘制可视化图片
    :param coord_w: x坐标列表
    :param coord_h: y坐标列表
    :param h: 标签阵列长
    :param w: 标签阵列宽
    :param path_fea: 可视化图片路径
    :return: None
    """
    fig = plt.figure()
    ax = fig.gca()
    # 将X坐标轴移到上面
    ax.xaxis.set_ticks_position('top')
    # 反转Y坐标轴
    ax.invert_yaxis()
    # matplotlib实现动态绘图
    # 打开交互模式
    plt.ion()
    last_w, last_h = coord_w[0], coord_h[0]
    distance = pow((h * 3) / 2, 2) + pow((w * 3) / 2, 2)
    for iter in range(len(coord_w)):
        if pow(coord_w[iter] - last_w, 2) + \
                pow(coord_w[iter] - last_w, 2) > distance:
            continue
        kalman_predict(coord_w[iter], coord_h[iter], iter, ax)
        last_w, last_h = coord_w[iter], coord_h[iter]
    # 关闭交互模式
    plt.ioff()
    # 去坐标轴
    plt.axis('off')
    # 去 x 轴刻度
    plt.xticks([])
    # 去 y 轴刻度
    plt.yticks([])
    plt.savefig(path_fea, bbox_inches='tight')


def kalman_predict(x, y, iter, ax):
    """
    使用kalman预测算法，平滑手指踪迹
    :param x: 初始坐标x
    :param y: 初始坐标y
    :param iter: 迭代次数
    :param ax: 可视化图片对象
    :return:
    """
    global filtered_state_means0, filtered_state_covariances0, lmx, lmy, lpx, lpy

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
        # 跟矩阵H对应着，Zk
        next_measurement = np.array([x, y])
        # next_measurement = np.array([x, y, z])
        cmx, cmy = x, y
        filtered_state_means, filtered_state_covariances = (
            kf.filter_update(filtered_state_means0, filtered_state_covariances0, next_measurement,
                             transition_offset=np.array([0, 0])))
        cpx, cpy = filtered_state_means[0][0], filtered_state_means[0][1]
        # 绘制真实轨迹和卡尔曼预测轨迹，红色是测量值，绿色是预测值
        ax.plot([lpx, cpx], [lpy, cpy], 'b', label='predict', linewidth=15.0)

        filtered_state_means0, filtered_state_covariances0 = filtered_state_means, filtered_state_covariances
        lpx, lpy = filtered_state_means[0][0], filtered_state_means[0][1]
        lmx, lmy = cmx, cmy


def del_file(path_data):
    """
    将原本路径下的文件清空
    :param path_data: 需清空的文件路径
    :return:
    """
    # os.listdir(path_data) 返回一个列表，里面是当前目录下面的所有东西的相对路径
    for i in os.listdir(path_data):
        # 当前文件夹的下面的所有东西的绝对路径
        file_data = path_data + "\\" + i
        # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
        if os.path.isfile(file_data):
            os.remove(file_data)
        else:
            del_file(file_data)


def data_to_csv(data: list, target: int):
    """
    将预处理后的数据保存成csv，作为train/test数据
    :param data: 预处理后的数据
    :param target: 手势
    :return: None
    """
    if os.path.exists(Path.PATH_DATA):
        df_old = pd.read_csv(Path.PATH_DATA)
    else:
        df_old = pd.DataFrame(columns=['feature', 'target'])
    target_list = [target] * len(data)
    df_new = pd.DataFrame(columns=['feature', 'target'])
    for data, target in zip(data, target_list):
        data_str = data.tolist()
        df_new = df_new.append({'feature': data_str, 'target': target}, ignore_index=True)
    df = pd.concat([df_old, df_new], ignore_index=True)
    df.to_csv(Path.PATH_DATA, index=False)


def split_train_test():
    """
    划分数据集为train and test
    :return:
    """
    if os.path.exists(Path.PATH_DATA):
        df = pd.read_csv(Path.PATH_DATA)
    else:
        raise NotADirectoryError
    features = df.iloc[:, 0].values
    target = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=Config.TEST_SIZE,
                                                        random_state=Config.RANDOM_SEED)

    # 创建训练集和测试集的DataFrame
    train_df = pd.DataFrame(X_train, columns=df.columns[:-1])
    train_df['target'] = y_train

    test_df = pd.DataFrame(X_test, columns=df.columns[:-1])
    test_df['target'] = y_test

    # 将训练集和测试集保存为CSV文件
    train_df.to_csv(Path.PATH_TRAIN, index=False)
    test_df.to_csv(Path.PATH_TEST, index=False)
