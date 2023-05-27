import os

from config.config import Config, Path
from utils.signal_process import (compute_likelihood, signal_preprocess, pant_img,
                                  segment_gesture, compute_reflect_signal,
                                  computer_theory_signal)
from utils.signal_prepare import together_tags, trans_to_numpy, part_tags

if __name__ == '__main__':

    # 当前目录下面的所有文件
    path_origin = os.listdir(Path.PATH_ORIGIN_GESTURE)

    for i in range(len(path_origin)):
        # 当前文件夹的下面的所有东西的绝对路径
        file_data = Path.PATH_ORIGIN_GESTURE + r'\\' + path_origin[i]
        Config.PATH_GESTURE_FEATURE = Path.PATH_GESTURE_FEATURE + \
            "\\" + path_origin[i][:-4]
        print("原始数据：", file_data)

        # 从原始数据中将每个标签数据（RSSI, Phase）提取到字典里，字典仅包含一个DataFrame（EPC, RSSI, Phase）
        # maxtime: 此数据中最晚的时间戳
        dict_data_move, maxtime_m = part_tags(
            path=file_data, tag_num=len(Config.TAG_ARRAY))

        # 每一个标签的数据都集合到一个工作表里（便于插值）
        together_tags(
            path=Path.PATH_TOGETHER,
            dic_data=dict_data_move,
            array=Config.TAG_ARRAY)

        # 将字典中的rssi, phase提取出来并转化为numpy数组形式
        # res_rssi[0][0] 代表坐标为(0, 0)的标签所有时刻的rssi值
        res_rssi, res_phase, res_time = trans_to_numpy(
            dic_data=dict_data_move, array=Config.TAG_ARRAY)

        # 所需参数为 原始信号值、该原始信号值的对应时间戳、该状态下读取信号的最大时间戳
        rssi_move = signal_preprocess(
            data=res_rssi, res_time=res_time, max_time=maxtime_m)
        phase_move = signal_preprocess(
            data=res_phase, res_time=res_time, max_time=maxtime_m)

        # rssi_static、phase_static 为rssi_move前多个时间戳的值，rssi_move、phase_move为最后能够直接使用的数据。
        # 通过实际测出信号计算实际反射信号
        actual_signal_array = compute_reflect_signal(
            rssi_move=rssi_move, phase_move=phase_move)
        # # 分割出 存在手势的时间段
        start, end = segment_gesture(phase_move=phase_move)

        # 计算手指在每个坐标上的理论反射信号, There_signal[0][0] 代表手指在坐标(0,0)的理论标签矩阵
        theory_signal_array = computer_theory_signal()

        # # 计算上述二者的相关系数
        like_hood, pearson = compute_likelihood(
            actual_signal_array=actual_signal_array,
            theory_signal_array=theory_signal_array)
        # # length 代表一共有多少个时刻，一个时刻一张图
        length = len(like_hood)
        # # 绘制手势时间段的手指踪迹
        save_tuple = (Path.PATH_GESTURE_FEATURE, i + 1, Config.GESTURE)
        pant_img(like_hood[start: end], start, save_tuple)
