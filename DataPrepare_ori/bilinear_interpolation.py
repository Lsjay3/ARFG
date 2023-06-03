# Jay的开发时间：2022/9/2  19:47
import cv2  # cv2 即opencv的库
import numpy as np  # 给numpy起别名np，该库Numerical Python是python的数学函数库


# 双线性插值实现
def bilinear_interpolation(img, out_dim):
    src_h, src_w, channels = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print("src_h,src_w= ", src_h, src_w)
    print("dst_h,dst_w= ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 根据几何中心重合找出目标像素的坐标
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                # 找出目标像素最邻近的四个点
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                if (src_x == src_x1 and src_x == src_x0):
                    print("src_x：", src_x)
                    print("src_x1：", src_x1)
                    print("src_x0：", src_x0)
                # 代入公式计算
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


import numpy as np
import math
import cv2


def double_linear(input_signal, zoom_multiples):
    '''
    双线性插值
    :param input_signal: 输入图像
    :param zoom_multiples: 放大倍数
    :return: 双线性插值后的图像
    '''
    input_signal_cp = np.copy(input_signal)  # 输入图像的副本

    input_row, input_col = input_signal_cp.shape  # 输入图像的尺寸（行、列）

    # 输出图像的尺寸
    output_row = int(input_row * zoom_multiples)
    output_col = int(input_col * zoom_multiples)

    output_signal = np.zeros((output_row, output_col))  # 输出图片

    for i in range(output_row):
        for j in range(output_col):
            # 输出图片中坐标 （i，j）对应至输入图片中的最近的四个点点（x1，y1）（x2, y2），（x3， y3），(x4，y4)的均值
            temp_x = i / output_row * input_row
            temp_y = j / output_col * input_col

            x1 = int(temp_x)
            y1 = int(temp_y)

            x2 = x1
            y2 = y1 + 1

            x3 = x1 + 1
            y3 = y1

            x4 = x1 + 1
            y4 = y1 + 1

            u = temp_x - x1
            v = temp_y - y1

            # 防止越界
            if x4 >= input_row:
                x4 = input_row - 1
                x2 = x4
                x1 = x4 - 1
                x3 = x4 - 1
            if y4 >= input_col:
                y4 = input_col - 1
                y3 = y4
                y1 = y4 - 1
                y2 = y4 - 1

            # 插值
            output_signal[i, j] = (1 - u) * (1 - v) * int(input_signal_cp[x1, y1]) + (1 - u) * v * int(
                input_signal_cp[x2, y2]) + u * (1 - v) * int(input_signal_cp[x3, y3]) + u * v * int(
                input_signal_cp[x4, y4])
    return output_signal

import numpy as np
import math

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
    return (f_00 * w_00).T + (f_01 * w_01).T + (f_10 * w_10).T + (f_11 * w_11).T

import matplotlib.pyplot as plt
if __name__ == '__main__':
    src = np.array([[1, 2, 3], [3, 4, 5], [6, 7, 8]])
    src = np.array([[0.03342193, 0.04013717, 0.05339826, 0.05220192, 0.04607979],
                    [0.02994744, 0.03739976, 0.0504893 , 0.0573916,  0.05343257],
                    [0.02596447, 0.03014549 ,0.04178158, 0.05239493 ,0.05214251],
                    [0.02351217, 0.0255327 , 0.03474959, 0.04508208 ,0.04595397],
                    [0.02430133 ,0.0263113 , 0.03336139, 0.03989176 ,0.04497497]])
    src = np.expand_dims(src, axis=2)
    print(src.shape)
    dst = bilinear_interpolate(src, dst_size=(src.shape[0] * 3, src.shape[1] * 3))
    print(dst.shape)
    # print(dst[:, :, 0])
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].matshow(src, cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3

    ax[1].matshow(dst[:, :, 0], cmap=plt.get_cmap('Blues'), alpha=0.5)  # , alpha=0.3
    plt.show()
# Read image

# img = cv2.imread("img385.png", 0).astype(np.float)
# print(img.shape)
# out = double_linear(img, 2).astype(np.uint8)
# print(out.shape)
# # Save result
# cv2.imshow("result", out)
# cv2.imwrite("out.jpg", out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# #
# img = cv2.imread("img385.png")
# print(img.shape)
# dst = bilinear_interpolation(img, (700, 700))
# cv2.imshow("blinear", dst)
# cv2.waitKey()
