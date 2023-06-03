# Jay的开发时间：2022/12/10  19:25
import numpy as np

import GetData
import DataPrepare
import os
import pandas as pd
import csv
import test
import matplotlib.pyplot as plt
from mtutils import bilinear_by_meshgrid
if __name__ == '__main__':
    # flag：此时处理的为静止0 or 手势1部分数据
    flag = 0
    # 标签摆放阵列

    col = ['E20000191107015510808551', 'E200001911070155085084F9', 'E200001911070154208080D1', 'E200001911070155084084F1', 'E2000019110701540370837D',
           'E200001911070155128085A1', 'E200001911070155079084ED', 'E200001911070155078084E5', 'E2000019110701550550848D', 'E200001911070154086082B5',
           'E20000191107015510208545', 'E200001911070154216080B1', 'E20000191107015418808121', 'E20000191107015419408105', 'E20000191107015503808445',
           'E20000191107015503008425', 'E20000191107015404308369', 'E2000019110701541930810D', 'E20000191107015416708179', 'E20000191107015410308279',
           'E200001911070154021083BD', 'E20000191107015413008205', 'E2000019110701541010827D', 'E20000191107015405808325', 'E20000191107015510108539']

    col = ['E20000191107015312207CD8', 'E20000191107015312307CE0', 'E20000191107015314407D24', 'E20000191107015315907D70', 'E20000191107015318607DD8',
           'E20000191107015311607CB4', 'E20000191107015313607D04', 'E20000191107015316407D74', 'E20000191107015317907DC0', 'E20000191107015318007DB4',
           'E20000191107015320507E1C', 'E20000191107015322507E6C', 'E20000191107015322607E78', 'E20000191107015324807EC4', 'E20000191107015326307F10',
           'E20000191107015320607E28', 'E20000191107015321907E60', 'E20000191107015324107EAC', 'E200001911070154209080CD', 'E2000019110701542370805D',
           'E20000191107015326407F04', 'E20000191107015328307F60', 'E20000191107015329007F78', 'E200001911070154217080AD', 'E20000191107015423108079']

    # col = ['E20000191107015309507C70', 'E20000191107015308107C2C', 'E20000191107015307507C20', 'E20000191107015310007C74', 'E20000191107015310207C88', 'E60000191107015309407C68',
    #        'E20000191107015312207CD8', 'E20000191107015312307CE0', 'E20000191107015314407D24', 'E20000191107015315907D70', 'E20000191107015318607DD8', 'E20000191107015308007C24',
    #        'E20000191107015311607CB4', 'E20000191107015313607D04', 'E20000191107015316407D74', 'E20000191107015317907DC0', 'E20000191107015318007DB4', 'E20000191107015305907BE0',
    #        'E20000191107015320507E1C', 'E20000191107015322507E6C', 'E20000191107015322607E78', 'E20000191107015324807EC4', 'E20000191107015326307F10', 'E20000191107015305207BB4',
    #        'E20000191107015320607E28', 'E20000191107015321907E60', 'E20000191107015324107EAC', 'E200001911070154209080CD', 'E2000019110701542370805D', 'E20000191107015306007BD4',
    #        'E20000191107015326407F04', 'E20000191107015328307F60', 'E20000191107015329007F78', 'E200001911070154217080AD', 'E20000191107015423108079', 'E20000191107015305107BC0']
    # col = ['35', '34', '33', '32', '31', '30', '29',
    #        '28', '27', '26', '25', '24', '23', '22',
    #        '21', '20', '19', '18', '17', '16', '15',
    #        '14', '13', '12', '11', '10', '9', '8',
    #        '7', '6', '5', '4', '3', '2', '1']

    width, height = 5, 5
    train_move = list()
    # 从原始数据中将每个标签数据（RSSI, Phase）提取到字典里，字典仅包含（EPC, RSSI, Phase）
    # maxtime:此数据中最晚的时间戳
    # path, dict_data_static, maxtime_s = GetData.Part_tags(r'E:\笨比J\RFID\Impinj R420\Taglogs\gesture_RotateLeft.csv', len(col), 0, 'E:\笨比J\RFID\Impinj R420\实验数据\二十五标签\二十五标签.xls')
    path_origin_data = r'E:\笨比J\RFID\Impinj R420\Data\原始数据\c\\'
    # path_feature = r'E:\笨比j\rfid\Impinj R420\Data\Train\swipe_left_to_right\\'

    # 最后特征图存放地址

    index = 1

    gesture_name = "c"
    path_ori = os.listdir(path_origin_data)
    for i in range(index - 1, len(path_ori)) : # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        path_feature = r'E:\笨比j\rfid\impinj r420\Data\Train\letters\c\\'
        file_data = path_origin_data + "\\" + path_ori[i]
        path_feature = path_feature + "\\" + path_ori[i][:-4]
        # print(path_feature)
        # file_data = r'E:\笨比J\RFID\Impinj R420\Taglogs\11-18-2022_12h_28m_28s.csv'
        # file_data = r'E:\笨比J\RFID\Impinj R420\Data\原始数据\c\\' + path_ori[index - 1] # 当前文件夹的下面的所有东西的绝对路径
        # path_feature = path_feature + "\\" + path_ori[index - 1][:-4]
        print("原始数据：", file_data)

        path, dict_data_move, maxtime_m = GetData.Part_tags(file_data, len(col), 1, 'E:\笨比J\RFID\Impinj R420\实验数据\二十五标签\二十五标签.xls')

        # 每一个标签的数据都集合到一个工作表里（便于插值）
        # GetData.Together_tags(r'E:\笨比J\RFID\Impinj R420\实验数据\二十五标签\二十五标签_together.xls', dict_data_static, flag, col)
        flag = 1
        GetData.Together_tags(r'E:\笨比J\RFID\Impinj R420\实验数据\二十五标签\二十五标签_together.xls', dict_data_move, flag, col)

        filter_win = 15
        expand = 3
        # 将字典中的rssi, phase提取出来并转化为numpy数组形式
        # res_rssi, res_phase, res_time = GetData.Transfer_np(dict_data_static, width, height, col)
        # res_rssi[0][0] 代表坐标为 (0, 0)的标签所有时刻的rssi值
        # 数据预处理：一维线性插值（时间上）、相位解缠绕、平滑滤波 所需参数为 原始信号值、该原始信号值的对应时间戳、该状态下读取信号的最大时间戳、滤波窗口
        # rssi_static = DataPrepare.Data_process(res_rssi, res_time, maxtime_s, filter_win)
        # phase_static = DataPrepare.Data_process(res_phase, res_time, maxtime_s, filter_win)

        res_rssi, res_phase, res_time = GetData.Transfer_np(dict_data_move, width, height, col)
        rssi_move = DataPrepare.Data_process(res_rssi, res_time, maxtime_m, filter_win)
        phase_move = DataPrepare.Data_process(res_phase, res_time, maxtime_m, filter_win)

        # rssi_static、phase_static、rssi_move、phase_move为最后能够直接使用的数据。
        # 通过实际测出信号计算实际反射信号
        # S_actual = DataPrepare.Compute_reflectsignal(rssi_static, phase_static, rssi_move, phase_move)
        S_actual = DataPrepare.Compute_reflectsignal(rssi_move, phase_move)
        # # 分割出 存在手势的时间段
        start, end = DataPrepare.Segment_gesture(phase_move)
        move = int((start + end) / 2)
        m = phase_move[:, :, move]
        move_list = []
        for i in range(height):
            for j in range(width):
                move_list.append(m[i][j])
        train_move.append(move_list)
        # if start == end:
        #     index += 1
        #     continue
        # 计算手指在每个坐标上的理论反射信号, There_signal[0][0] 代表手指在坐标(0,0)的理论标签矩阵
        There_signal = DataPrepare.Computer_theresignal(height, width)

        # # 计算上述二者的相关系数
        like_hood, pearson = DataPrepare.Compute_likelihood(S_actual, There_signal)
        # # length 代表一共有多少个时刻，一个时刻一张图
        length = len(like_hood)
        # save_tuple = (path_feature, index, gesture_name)
        # # 绘制整个时间段的手势踪迹
        # DataPrepare.Pant_img(like_hood[0: length], 0, height, width, save_tuple)
        # print("输入开始与结束:")
        # start, end = map(int, input().split())
        start = start
        end = end
        save_tuple = (path_feature, index, gesture_name)
        # DataPrepare.Pant_img(like_hood[start: end], start, height, width, save_tuple)

        # 将运动中的手势分割成开始、中间、结束三个阶段，并将三张图片组合在一起作为模型输入
        # save_tuple = (path_feature, index, gesture_name)
        # featrueImg = DataPrepare.Get_recombination(pearson, start, end, height, width, save_tuple)
        with open('test.csv', 'w') as file:
            writer = csv.writer(file)
        data = pd.DataFrame(train_move)
        data.to_csv('test.csv')
        index += 1



