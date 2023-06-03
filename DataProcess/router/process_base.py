# Jay的开发时间：2023/6/3  11:52
import os
import numpy

from config.config import Config, Path
from utils.signal_process import (compute_likelihood, signal_preprocess,
                                  segment_gesture, compute_reflect_signal,
                                  computer_theory_signal)
from utils.signal_prepare import trans_to_numpy, part_tags


class ProcessBase:
    def __init__(self, gesture):
        self.gesture = gesture

    def together_tags(self, dic_data: dict, array: str):
        pass

    def pant_img(self, like_hood: numpy.ndarray, start: int, save_tuple: tuple):
        pass

    def data_to_csv(self, data: list, target: str):
        pass

    def process(self):
        train_list = []
        path_origin_gesture = Path.PATH_ORIGIN_DATA + self.gesture
        origin_data = os.listdir(path_origin_gesture)
        for i in range(len(origin_data)):
            # 当前文件夹的下面的所有东西的绝对路径
            file_data = path_origin_gesture + r'\\' + origin_data[i]

            # 特征图存储路径
            Config.PATH_GESTURE_FEATURE = Path.PATH_GESTURE_FEATURE + \
                "\\" + origin_data[i][:-4]
            print("原始数据：", file_data)

            # 从原始数据中将每个标签数据（RSSI, Phase）提取到字典里，字典仅包含一个DataFrame（EPC, RSSI, Phase）
            # maxtime: 此数据中最晚的时间戳
            dict_data_move, maxtime_m = part_tags(
                path=file_data, tag_num=len(Config.TAG_ARRAY))

            # 每一个标签的数据都集合到一个工作表里（便于插值）
            self.together_tags(
                dic_data=dict_data_move,
                array=Config.TAG_ARRAY)
            # print("原始数据写入(手势状态)已完成")

            # 将字典中的rssi, phase提取出来并转化为numpy数组形式
            # res_rssi[0][0] 代表坐标为(0, 0)的标签所有时刻的rssi值
            res_rssi, res_phase, res_time = trans_to_numpy(
                dic_data=dict_data_move, array=Config.TAG_ARRAY)
            print("rssi、phase提取完成")

            # 所需参数为 原始信号值、该原始信号值的对应时间戳、该状态下读取信号的最大时间戳
            rssi_move = signal_preprocess(
                data=res_rssi, res_time=res_time, max_time=maxtime_m)
            phase_move = signal_preprocess(
                data=res_phase, res_time=res_time, max_time=maxtime_m)
            print("数据预处理完成")

            # rssi_static、phase_static 为rssi_move前多个时间戳的值，rssi_move、phase_move为最后能够直接使用的数据。
            # 通过实际测出信号计算实际反射信号
            actual_signal_array = compute_reflect_signal(
                rssi_move=rssi_move, phase_move=phase_move)
            print("反射信号计算完成")
            # # 分割出 存在手势的时间段
            start, end = segment_gesture(phase_move=phase_move)

            # 计算手指在每个坐标上的理论反射信号, There_signal[0][0] 代表手指在坐标(0,0)的理论标签矩阵
            theory_signal_array = computer_theory_signal()
            print("理论信号计算完成")

            # # 计算上述二者的相关系数
            like_hood, pearson = compute_likelihood(
                actual_signal_array=actual_signal_array,
                theory_signal_array=theory_signal_array)
            print("似然估计量计算完成")
            # # length 代表一共有多少个时刻，一个时刻一张图
            length = len(like_hood)
            # # 绘制手势时间段的手指踪迹
            save_tuple = (Path.PATH_GESTURE_FEATURE, i + 1, Config.GESTURE)
            gesture_stream = like_hood[start: end]
            train_list.append(gesture_stream)
            self.pant_img(like_hood[start: end], start, save_tuple)

        self.data_to_csv(data=train_list, target=Config.GESTURE_DICT[self.gesture])
