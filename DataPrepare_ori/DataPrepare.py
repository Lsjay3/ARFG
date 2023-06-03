# Jay的开发时间：2022/8/3  10:15
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import xlwt
import numpy as np
from pykalman import KalmanFilter
from scipy import interpolate
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.stats import pearsonr

import FileUtils
import ImgUtils


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
    return np.convolve(a, np.ones((n,)) / n, mode=mode)


def get_variance_stream(y, window_size):
    """
    求给定数组的各滑动窗口方差
    :param y: 要求滑动方差的数组
    :param window_size: 滑动窗口大小
    :return:
    """
    v = []
    for i in range(len(y)):
        if i + window_size >= len(y):
            break
        else:
            gap = window_size
        v.append(np.var(y[i: i + gap]))
    return v


def segment_gesture(data: np.ndarray, variance_img_save_path: str):
    """
    根据给定的标签阵列相位数据获取手势起始点索引
    :param variance_img_save_path: 方差图保存位置，为空则不保存
    :param data: 给定标签阵列的相位数据
    :return: 手势起点和终点
    """
    tags_array_height, tags_array_width, channel = data.shape
    variance = list()
    for i in range(tags_array_height):
        for j in range(tags_array_width):
            v = get_variance_stream(data[i][j], int(channel / 30))
            variance.append(v)
    variance = np.array(variance)
    variance_max = list()
    # 求在每个滑动窗口中所有标签的最大方差
    for i in range(len(variance[0])):
        variance_max.append(np.max(variance[:, i]))
    # 求峰值
    min_height = np.mean(variance_max)
    peaks, _ = find_peaks(variance_max, height=min_height)
    if len(peaks) <= 1:
        return 0, 0
    print("手势起始点：", peaks[0], peaks[-1])
    # 绘图
    if variance_img_save_path == "":
        return peaks[0], peaks[-1]
    x = list(np.arange(1, len(variance[0]) + 1))
    plt.plot(x, variance_max, lw=4, ls='-', c='k', alpha=0.5)
    plt.savefig(variance_img_save_path)
    #plt.show()
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
    x = np.linspace(0, orig_w, orig_w, endpoint=False)
    y = np.linspace(0, orig_h, orig_h, endpoint=False)
    xx = np.linspace(0, orig_w, orig_w * n, endpoint=False)
    yy = np.linspace(0, orig_h, orig_h * n, endpoint=False)
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


def get_signal(rssi_static, filter_unwrap_phase_static, rssi_move, filter_unwrap_phase_move):
    """
    使用欧拉公式计算信号
    :param rssi_static:
    :param filter_unwrap_phase_static:
    :param rssi_move:
    :param filter_unwrap_phase_move:
    :return:
    """
    drssi_static = pow(pow(10, (rssi_static / 10 - 3)), 0.5)
    drssi_move = pow(pow(10, (rssi_move / 10 - 3)), 0.5)
    cos_phase_static = np.math.cos(filter_unwrap_phase_static)
    sin_phase_static = np.math.sin(filter_unwrap_phase_static)
    cos_phase_move = np.math.cos(filter_unwrap_phase_move)
    sin_phase_move = np.math.sin(filter_unwrap_phase_move)

    actual_signal = pow((drssi_move * cos_phase_move) - (drssi_static * cos_phase_static), 2) + pow(
        (drssi_move * sin_phase_move) - (drssi_static * sin_phase_static), 2)
    return actual_signal


def compute_reflect_signal(rssi_move: np.ndarray, phase_move: np.ndarray, save_signal_data_to_path: str = "",
                           save_signal_images_to_path: str = ""):
    """
    通过实际测出信号计算实际反射信号
    :param save_signal_images_to_path: 根据计算结果生成的图像保存位置，为空不保存
    :param save_signal_data_to_path: 计算结果保存位置，为空则不保存
    :param rssi_move: RSSI数据
    :param phase_move: 相变数据
    :return:
    """
    tags_array_height, tags_array_width, channels = rssi_move.shape
    actual_signal = np.zeros((tags_array_height, tags_array_width, channels), dtype=np.float64)
    for i in range(tags_array_height):
        for j in range(tags_array_width):
            rssi_static = np.mean(rssi_move[i][j][0:10])
            phase_static = np.mean(phase_move[i][j][0:10])
            for channel_index in range(channels):
                # 计算坐标（i + 1, j + 1）在 t 时刻的反射信号强度
                actual_signal[i][j][channel_index] = get_signal(0, phase_static,
                                                                rssi_move[i][j][channel_index] - rssi_static,
                                                                phase_move[i][j][channel_index])
    print("反射信号计算完成")

    if not save_signal_data_to_path == '':
        return actual_signal

    # 检查并创建路径
    FileUtils.create_directory_if_not_exist(save_signal_data_to_path)
    # for channel_index in range(channels):
    #     im = plt.matshow(actual_signal[:, :, channel_index], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
    #     plt.show()
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('Reflect_signal', cell_overwrite_ok=True)
    for i in range(1, tags_array_height * tags_array_width + 1):
        sheet.write(0, i, str(i))
    i = 1
    for height_index in range(tags_array_height):
        for width_index in range(tags_array_width):
            for j, data in enumerate(actual_signal[height_index][width_index]):
                sheet.write(j + 1, i, data)
            i += 1
    book.save(save_signal_data_to_path)
    print("反射信号计算结果已保存至：" + save_signal_data_to_path)

    if not save_signal_images_to_path == '':
        return actual_signal

    # 检查并创建路径
    FileUtils.create_directory_if_not_exist(save_signal_images_to_path)
    images_path = Path(save_signal_images_to_path)
    col = int(np.math.sqrt(channels))
    row = int(np.math.sqrt(channels))
    for channel_index in range(channels):
        plt.matshow(actual_signal[:, :, channel_index], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
        plt.title(channel_index)
        plt.savefig(images_path / f'img{channel_index}.jpg')
        plt.close()
    ImgUtils.compose_images(save_signal_images_to_path, str(images_path / 'all.jpg'), row, col)
    print("反射信号图片已保存至：" + save_signal_images_to_path)
    return actual_signal


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
    for t in range(channels):  # 某一时刻 共283个时刻
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
    new_matrix = [total_element[x:x + len(matrix1)] for x in range(0, len(total_element), len(matrix1))]
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
    ImgUtils.del_file(save_path)
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

    likeli = np.zeros((h, w, 3), dtype=np.float64)
    likeli[:, :, 0] = likeli_start
    likeli[:, :, 1] = likeli_middle
    likeli[:, :, 2] = likeli_end
    # 对三个时期矩阵进行双线性插值增加分辨率
    feature = bilinear_interpolation(likeli, 3)

    # 将三个矩阵进行垂直拼接
    tmp = np.concatenate([feature[:, :, 0], feature[:, :, 1]], axis=0)
    res_like = np.concatenate([tmp, feature[:, :, 2]], axis=0)
    im = plt.matshow(feature[:, :, 0], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去 x 轴刻度
    plt.yticks([])  # 去 y 轴刻度
    plt.savefig(r'E:\笨比j\rfid\Impinj R420\img_feature\1.jpg', bbox_inches='tight', pad_inches=-0.1)
    plt.close()
    im = plt.matshow(feature[:, :, 1], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去 x 轴刻度
    plt.yticks([])  # 去 y 轴刻度
    plt.savefig(r'E:\笨比j\rfid\Impinj R420\img_feature\2.jpg', bbox_inches='tight', pad_inches=-0.1)
    plt.close()
    im = plt.matshow(feature[:, :, 2], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去 x 轴刻度
    plt.yticks([])  # 去 y 轴刻度
    plt.savefig(r'E:\笨比j\rfid\Impinj R420\img_feature\3.jpg', bbox_inches='tight', pad_inches=-0.1)
    plt.close()
    im = plt.matshow(res_like, cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去 x 轴刻度
    plt.yticks([])  # 去 y 轴刻度
    plt.colorbar(im)
    plt.close()
    # Img_compose.image_compose(save_path, r'E:\笨比j\rfid\Impinj R420\img_feature\feature.jpg', 3, 1)
    ImgUtils.compose_images(save_path, path_fea, 3, 1)
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


def linear_interpolation(rssi_data: np.ndarray, rssi_time: np.ndarray, max_time: int) -> np.ndarray:
    """
    对RSSI数据进行线性插值
    :param rssi_data: 待插值的RSSI数据
    :param rssi_time: 待插值RSSI对应的事件戳序列
    :param max_time: 事件戳中最大的值
    :return: 插值后的RSSI数据
    """
    x_val = np.arange(0, max_time, 40)
    return np.interp(x_val, rssi_time, rssi_data)


def data_process(rssi_data: np.ndarray, rssi_time: np.ndarray, max_time: int, windows_size: int,
                 save_to_path: str) -> np.ndarray:
    """
    数据预处理：一维线性插值（时间上）、相位解缠绕、平滑滤波
    :param save_to_path: 处理后的数据保存路径，为空则不保存
    :param rssi_data: 原始信号值
    :param rssi_time: 原始信号值的对应时间戳
    :param max_time: 该状态下读取信号的最大时间戳
    :param windows_size: 滤波窗口大小
    :return:
    """
    tags_array_height, tags_array_width, channels = rssi_data.shape

    # 进行一维线性插值（时间上的插值，解决标签与天线的随机通信），间隔40ms插一次值
    channel = int(max_time / 40) + 1
    res = np.zeros((tags_array_height, tags_array_width, channel), dtype=np.float64)

    for i in range(tags_array_height):
        for j in range(tags_array_width):
            # 将相位进行解缠绕处理后计算线性插值
            if rssi_data[0][0][0] > -10:
                res[i][j] = linear_interpolation(np.unwrap(rssi_data[i][j]), rssi_time[i][j], max_time)
            else:
                res[i][j] = linear_interpolation(rssi_data[i][j], rssi_time[i][j], max_time)
            # 进行滤波操作
            res[i][j] = savgol_filter(res[i][j], windows_size, 2)
    print("数据预处理完成")

    if not save_to_path == '':
        return res

    # 将处理后的数据写入表格
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('final_data', cell_overwrite_ok=True)
    for i in range(1, tags_array_height * tags_array_width + 1):
        sheet.write(0, i, str(i))
    i = 1
    for h in range(tags_array_height):
        for w in range(tags_array_width):
            for j, rssi_data in enumerate(res[h][w]):
                sheet.write(j + 1, i, rssi_data)
            i += 1
    book.save(save_to_path)
    print("预处理后的数据已保存到 " + save_to_path + " 中")
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


def Pant_img(like_hood, s, h, w, save_tuple, save_to_path):
    FileUtils.create_directory_if_not_exist(save_to_path)
    path_feature, index, gesture_name = save_tuple
    path_fea = str(Path(path_feature) / f"{gesture_name}.{index}.jpg")
    l = len(like_hood)
    k = 5
    # 计算 一行与一列最大容纳多少张图片（图片合并）
    col = int(np.math.sqrt(l))
    row = int(np.math.sqrt(l))
    data = Trans_likeToImg(like_hood, h, w)
    # 进行双线性插值，增加图片分辨率
    data = bilinear_interpolation(data, 3)
    ImgUtils.del_file(save_to_path)
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
        plt.savefig(save_to_path + r'\img' + str(s + t) + '.jpg')
        # plt.colorbar(im)
        plt.close()
    P_trace(coord_w, coord_h, h, w, path_fea)
    # Pant_trace(coord_trace, h, w, path_fea)
    ImgUtils.compose_images(save_to_path, save_to_path + r'\all.jpg', row, col)
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
