from pathlib import Path

import DataPrepare
import FileUtils
import GetData

if __name__ == '__main__':
    # 标签摆放阵列

    tags_array = ['E20000191107015510808551', 'E200001911070155085084F9', 'E200001911070154208080D1',
                  'E200001911070155084084F1', 'E2000019110701540370837D',
                  'E200001911070155128085A1', 'E200001911070155079084ED', 'E200001911070155078084E5',
                  'E2000019110701550550848D', 'E200001911070154086082B5',
                  'E20000191107015510208545', 'E200001911070154216080B1', 'E20000191107015418808121',
                  'E20000191107015419408105', 'E20000191107015503808445',
                  'E20000191107015503008425', 'E20000191107015404308369', 'E2000019110701541930810D',
                  'E20000191107015416708179', 'E20000191107015410308279',
                  'E200001911070154021083BD', 'E20000191107015413008205', 'E2000019110701541010827D',
                  'E20000191107015405808325', 'E20000191107015510108539']

    tags_array = ['E20000191107015312207CD8', 'E20000191107015312307CE0', 'E20000191107015314407D24',
                  'E20000191107015315907D70', 'E20000191107015318607DD8',
                  'E20000191107015311607CB4', 'E20000191107015313607D04', 'E20000191107015316407D74',
                  'E20000191107015317907DC0', 'E20000191107015318007DB4',
                  'E20000191107015320507E1C', 'E20000191107015322507E6C', 'E20000191107015322607E78',
                  'E20000191107015324807EC4', 'E20000191107015326307F10',
                  'E20000191107015320607E28', 'E20000191107015321907E60', 'E20000191107015324107EAC',
                  'E200001911070154209080CD', 'E2000019110701542370805D',
                  'E20000191107015326407F04', 'E20000191107015328307F60', 'E20000191107015329007F78',
                  'E200001911070154217080AD', 'E20000191107015423108079']

    tags_array_width, tags_array_height = 5, 5

    # flag：此时处理的为静止0 or 手势1部分数据
    flag = 1

    # 原始数据总存放目录
    origin_data_directory = Path.cwd() / '原始数据'
    # 输出目录
    output_directory = Path.cwd() / 'Output'
    # 手势名称
    gesture_name = "c"
    # 特征图总输出目录
    feature_directory = output_directory / 'FeaturesImages' / 'Letters'
    # 原始数据转excel文件输出目录
    original_data_excel_directory = output_directory / 'OriginalDataExcel'
    # 处理后的数据保存目录
    processed_data_excel_directory = output_directory / 'ProcessedDataExcel'
    # 反射信号数据保存目录
    reflect_signal_data_directory = output_directory / 'ReflectSignal' / 'Data'
    # 反射信号图片保存目录
    reflect_signal_images_directory = output_directory / 'ReflectSignal' / 'Images'
    # 方差图片保存目录
    variance_img_directory = output_directory / 'Variance'
    # 最终分割出的图片保存目录
    segmented_images_directory = output_directory / 'SegmentedPicture'
    # 是否需要输出excel文件
    output_to_excel = True

    # 指定文件
    specific_file = '11-16-2022_21h_01m_10s.csv'

    # 原始数据目录
    origin_data_path = origin_data_directory / gesture_name
    # 特征图输出目录
    feature_path = feature_directory / gesture_name

    original_file_list = list()
    # 若指定了文件则只对指定的文件进行转换
    if specific_file != '':
        original_file_list.append(str(origin_data_path / specific_file))
    else:
        for origin_file in origin_data_path.iterdir():
            original_file_list.append(str(origin_file))

    # 检查并创建输出目录
    FileUtils.create_directory_if_not_exist(str(feature_path))
    FileUtils.create_directory_if_not_exist(str(original_data_excel_directory))
    FileUtils.create_directory_if_not_exist(str(processed_data_excel_directory))
    FileUtils.create_directory_if_not_exist(str(variance_img_directory))

    print()
    print("原始数据列表：", original_file_list)

    for original_file_index, original_file in enumerate(original_file_list):
        original_file_name = Path(original_file).stem
        excel_path = str(original_data_excel_directory / f'{original_file_name}.xls')
        path, dict_data_move, max_timestamp = GetData.part_tags(original_file, len(tags_array), flag, excel_path,
                                                                output_to_excel)

        # 每一个标签的数据都集合到一个工作表里（便于插值）
        # GetData.Together_tags(r'.\实验数据\二十五标签\二十五标签_together.xls', dict_data_static, flag, col)
        # GetData.Together_tags(r'.\实验数据\二十五标签\二十五标签_together.xls', dict_data_move, flag, col)

        filter_window_size = 15
        expand = 3

        res_rssi, res_phase, res_time = GetData.transfer_to_np(dict_data_move, tags_array_width, tags_array_height,
                                                               tags_array)
        rssi_move = DataPrepare.data_process(res_rssi, res_time, max_timestamp, filter_window_size,
                                             str(processed_data_excel_directory))
        phase_move = DataPrepare.data_process(res_phase, res_time, max_timestamp, filter_window_size,
                                              str(processed_data_excel_directory))

        actual_signal = DataPrepare.compute_reflect_signal(rssi_move, phase_move,
                                                           str(reflect_signal_data_directory
                                                               / f'{original_file_name}.xls'),
                                                           str(reflect_signal_images_directory / original_file_name))
        # 分割出 存在手势的时间段
        start, end = DataPrepare.segment_gesture(phase_move, variance_img_save_path=str(
            variance_img_directory / f'{original_file_name}.jpg'))
        if start == end:
            print(f"未成功分割出{original_file_name}的手势，已跳过此样本的后续处理")
            continue
        # 计算手指在每个坐标上的理论反射信号, There_signal[0][0] 代表手指在坐标(0,0)的理论标签矩阵
        There_signal = DataPrepare.Computer_theresignal(tags_array_height, tags_array_width)

        # 计算上述二者的相关系数
        like_hood, pearson = DataPrepare.Compute_likelihood(actual_signal, There_signal)
        # length 代表一共有多少个时刻，一个时刻一张图
        length = len(like_hood)
        save_tuple = (str(feature_path), original_file_index, gesture_name)
        # 绘制整个时间段的手势踪迹
        # DataPrepare.Pant_img(like_hood[0: length], 0, tags_array_height, tags_array_width, save_tuple)
        # print("输入开始与结束:")
        # start, end = map(int, input().split())
        # start = start
        # end = end
        DataPrepare.Pant_img(like_hood[start: end], start, tags_array_height, tags_array_width, save_tuple,
                             str(segmented_images_directory))

        # 将运动中的手势分割成开始、中间、结束三个阶段，并将三张图片组合在一起作为模型输入
        # save_tuple = (path_feature, index, gesture_name)
        # feature_Img = DataPrepare.Get_recombination(pearson, start, end, height, width, save_tuple)
        original_file_index += 1
