# Jay的开发时间：2022/8/29  9:53
import datetime
import time

import numpy as np
import pandas as pd
import xlrd
from config.config import Config
from xlutils.copy import copy


def part_tags(path: str, tag_num: int):
    """
    从原始数据中将每个标签数据（RSSI, Phase）提取到字典里，字典仅包含（EPC, RSSI, Phase）
    :param path: 原始数据路径，exp：‘E:\\XXX\\RFID\\Impinj R420\\Data\\origin_data\\c\11-16-2022_20h_58m_43s.csv’
    :param tag_num: 标签个数
    :return:
    dict_data = {
        "E20000191107015322607E78": a DataFrame 包含 Timestamp, EPC, RSSI, Phase
    }
    max_timestamp: 这批数据最晚的时间戳
    """
    data = pd.read_csv(
        path,
        usecols=[
            '// Timestamp',
            ' EPC',
            ' RSSI',
            ' PhaseAngle'])
    if isinstance(data.iloc[1, 0], str):
        time_start = datetime.datetime.strptime(
            data.iloc[0, 0][:-7], "%Y-%m-%dT%H:%M:%S.%f")
        start_stamp = int(
            time.mktime(
                time_start.timetuple()) * 1000000.0 + time_start.microsecond)
        max_timestamp = 0
        for i in range(len(data)):
            new_t = datetime.datetime.strptime(
                data.iloc[i, 0][:-7], "%Y-%m-%dT%H:%M:%S.%f")
            obj_stamp = int(
                time.mktime(
                    new_t.timetuple()) * 1000000.0 + new_t.microsecond)
            data.iloc[i, 0] = (obj_stamp - start_stamp) / 1000.0
            if data.iloc[i, 0] > max_timestamp:
                max_timestamp = data.iloc[i, 0]
    # 加入读入的为另外一种数据，时间戳形式为：824953
    else:
        max_timestamp = 0
        time_start = data.iloc[0, 0]
        for i in range(len(data)):
            data.iloc[i, 0] = (data.iloc[i, 0] - time_start)
            if data.iloc[i, 0] > max_timestamp:
                max_timestamp = data.iloc[i, 0]
    Tags_name = list()
    Tags = list()

    i = 0
    # 获得标签名字
    while len(Tags_name) < tag_num:
        if data.iloc[i, 1] not in Tags_name:
            Tags_name.append(str(data.iloc[i, 1]))
        i += 1
    for names in Tags_name:
        # 筛选不同标签数据，并和其EPC一一对应
        Tags.append(data.loc[data[' EPC'] == names, :])
    dic_data = dict(zip(Tags_name, Tags))
    return dic_data, max_timestamp


def together_tags(path: str, dic_data: dict, array: list):
    """
    将所有标签数据写入一个任务表sheet中，用于查看数据变化及插值
    :param path: 目标任务表路径
    :param dic_data: 以EPC为主键，DataFrame为值的字典
    :param array: 标签阵列（一维）
    :return: None
    """
    book = xlrd.open_workbook(path)
    book_prepare = copy(book)
    sheet_name = ['RSSI_手势', 'Phase_手势']
    for sheetnames in sheet_name:
        if sheetnames in book.sheet_names():
            sheet = book_prepare.get_sheet(sheetnames)
        else:
            sheet = book_prepare.add_sheet(
                sheetnames, cell_overwrite_ok=True)  # 创建一张表单
        for j in range(len(array)):
            sheet.write(0, j, array[j])
            data = np.zeros((Config.HEIGHT, Config.WIDTH), dtype=np.float64)
            if sheetnames == 'RSSI_静止' or sheetnames == 'RSSI_手势':
                data = dic_data[array[j]].iloc[:, 2].to_numpy()
            elif sheetnames == 'Phase_静止' or sheetnames == 'Phase_手势':
                data = np.unwrap(dic_data[array[j]].iloc[:, 3].to_numpy())
            for i in range(1, len(data) + 1):
                sheet.write(i, j, str(data[i - 1]))
    print("原始数据写入(手势状态)已完成")
    book_prepare.save(path)


def trans_to_numpy(dic_data: dict, array: list):
    """
    将原始标签数据转化为numpy数组形式（height * width * num）
    :param dic_data: 以EPC为主键，DataFrame为值的字典
    :param array: 标签阵列（一维）
    :return:
    res_rssi[0][0] 代表坐标为(0, 0)的标签所有时刻的rssi值
    res_phase[0][0] 代表坐标为(0, 0)的标签所有时刻的phase值
    res_time 同理
    """
    length = len(dic_data[array[0]])
    for i in range(Config.WIDTH * Config.HEIGHT):
        if len(dic_data[array[i]]) < length:
            length = len(dic_data[array[i]])
    res_rssi = np.zeros((Config.HEIGHT, Config.WIDTH, length), dtype=np.float64)
    res_phase = np.zeros((Config.HEIGHT, Config.WIDTH, length), dtype=np.float64)
    res_time = np.zeros((Config.HEIGHT, Config.WIDTH, length), dtype=np.float64)
    index = 0
    for i in range(Config.HEIGHT):
        for j in range(Config.WIDTH):
            data_time = dic_data[array[index]].iloc[:, 0].to_numpy()
            data_rssi = dic_data[array[index]].iloc[:, 2].to_numpy()
            data_phase = dic_data[array[index]].iloc[:, 3].to_numpy()
            for k in range(len(data_rssi)):
                if length <= k:
                    continue
                res_rssi[i][j][k] = data_rssi[k]
                res_phase[i][j][k] = data_phase[k]
                res_time[i][j][k] = data_time[k]
            index += 1
    print("rssi、phase提取完成")
    return res_rssi, res_phase, res_time
