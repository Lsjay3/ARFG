# Jay的开发时间：2022/8/29  9:53
import datetime
import os.path
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import xlrd
import xlwt
from pandas import DataFrame
from xlutils.copy import copy


# todo: 测试是否和原本的行为一致
def together_tags(path: str, dic_data: Dict[str, DataFrame], flag: int, col: list):
    """
    将所有标签数据写入一个任务表sheet中
    :param path: 保存位置
    :param dic_data: 标签数据
    :param flag: 静止 0 / 手势 1
    :param col: 表头
    :return: 无
    """
    book = xlrd.open_workbook(path)
    book_prepare: xlwt.Workbook = copy(book)
    if flag == 0:
        sheet_names = ['RSSI_静止', 'Phase_静止']
    else:
        sheet_names = ['RSSI_手势', 'Phase_手势']
    for sheet_name in sheet_names:
        if sheet_name in book.sheet_names():
            sheet: xlwt.Worksheet = book_prepare.get_sheet(sheet_name)
        else:
            sheet: xlwt.Worksheet = book_prepare.add_sheet(sheet_name, cell_overwrite_ok=True)  # 创建一张表单
        for j in range(len(col)):
            # 写入表头
            sheet.write(0, j, col[j])
            if sheet_name == 'RSSI_静止' or sheet_name == 'RSSI_手势':
                data = dic_data[col[j]].iloc[:, 2].to_numpy()
            elif sheet_name == 'Phase_静止' or sheet_name == 'Phase_手势':
                data = dic_data[col[j]].iloc[:, 3].to_numpy()
            else:
                continue
            for i in range(1, len(data) + 1):
                sheet.write(i, j, str(data[i - 1]))
    if flag == 0:
        print("原始数据写入(静止状态)已完成")
    else:
        print("原始数据写入(手势状态)已完成")
    book_prepare.save(path)


def part_tags(path: str, tag_num: int, flag: int, path_store: str, write_to_excel=True) \
        -> (str, Dict[str, DataFrame], int):
    """
    从原始csv数据中将每个标签数据（RSSI, Phase）提取到字典里，字典仅包含（EPC, RSSI, Phase）
    并可选地保存到指定的excel文件中
    :param write_to_excel: 是否将数据保存到存储路径的表格中
    :param path: 数据路径
    :param tag_num: 标签个数
    :param flag: 静止 0 / 手势 1
    :param path_store: 存储路径
    :return: Excel文件最终保存路径，原始数据，数据中的最大相对时间戳
    """
    data = pd.read_csv(path, usecols=['// Timestamp', ' EPC', ' RSSI', ' PhaseAngle'])
    max_timestamp = 0
    # 将时间戳转换为相对时间戳（以第一条数据为起点）
    if isinstance(data.iloc[1, 0], str):
        time_start = datetime.datetime.strptime(data.iloc[0, 0][:-7], "%Y-%m-%dT%H:%M:%S.%f")
        start_stamp = int(time.mktime(time_start.timetuple()) * 1000000.0 + time_start.microsecond)

        for i in range(len(data)):
            new_t = datetime.datetime.strptime(data.iloc[i, 0][:-7], "%Y-%m-%dT%H:%M:%S.%f")
            obj_stamp = int(time.mktime(new_t.timetuple()) * 1000000.0 + new_t.microsecond)
            data.iloc[i, 0] = (obj_stamp - start_stamp) / 1000.0
            if data.iloc[i, 0] > max_timestamp:
                max_timestamp = data.iloc[i, 0]
    else:
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

    # 按EPC分组
    for names in Tags_name:
        Tags.append(data.loc[data[' EPC'] == names, :])
    dic_data = dict(zip(Tags_name, Tags))

    if not write_to_excel:
        return path_store, dic_data, max_timestamp

    excel_directory_path = Path(path_store).parent
    if not excel_directory_path.exists():
        os.makedirs(str(excel_directory_path))
    # 写入工作表
    book = xlwt.Workbook(encoding="utf8")
    col = ["静止时间戳", "RSSI_静止", "Phase_静止", "手势时间戳", "RSSI_手势", "Phase_手势"]
    for names in Tags_name:
        sheet = book.add_sheet(names, cell_overwrite_ok=True)  # 创建一张表单
        timestamp = dic_data[names].iloc[:, 0].to_numpy()
        rssi = dic_data[names].iloc[:, 2].to_numpy()
        phase = dic_data[names].iloc[:, 3].to_numpy()
        for j in range(len(col)):
            sheet.write(0, j, col[j])
        for j in range(1, len(rssi) + 1):
            if flag == 0:
                sheet.write(j, 0, str(timestamp[j - 1]))
                sheet.write(j, 1, str(rssi[j - 1]))
                sheet.write(j, 2, str(phase[j - 1]))
            else:
                sheet.write(j, 3, str(timestamp[j - 1]))
                sheet.write(j, 4, str(rssi[j - 1]))
                sheet.write(j, 5, str(phase[j - 1]))
    book.save(path_store)
    if flag == 0:
        print("标签分离(静止状态)已完成")
    else:
        print("标签分离(手势状态)已完成")
    return path_store, dic_data, max_timestamp


def transfer_to_np(dic_data: Dict[str, DataFrame], tags_array_width: int, tags_array_height: int, tags_array: list):
    """
    将原始标签数据转化为numpy数组形式，返回数据在数组中的位置与在标签阵列中的位置相对应，例如：return_data_rssi[0][0] 代表坐标为 (0, 0)的标签所有时刻的rssi值
    :param dic_data: 原始标签数据
    :param tags_array_width: 标签阵列宽度（单位：个）
    :param tags_array_height: 标签阵列高度（单位：个）
    :param tags_array: 标签名（按标签阵列自左向右自上而下排列）
    :return: 标签阵列中各标签采集到的RSSI、相位、时间戳数据，形状为：height, width, num。
    """
    useful_rows_length = len(dic_data[tags_array[0]])
    for i in range(tags_array_width * tags_array_height):
        if len(dic_data[tags_array[i]]) < useful_rows_length:
            useful_rows_length = len(dic_data[tags_array[i]])
    res_rssi = np.zeros((tags_array_height, tags_array_width, useful_rows_length), dtype=np.float64)
    res_phase = np.zeros((tags_array_height, tags_array_width, useful_rows_length), dtype=np.float64)
    res_time = np.zeros((tags_array_height, tags_array_width, useful_rows_length), dtype=np.float64)
    index = 0
    for i in range(tags_array_height):
        for j in range(tags_array_width):
            data_time = dic_data[tags_array[index]].iloc[:, 0].to_numpy()
            data_rssi = dic_data[tags_array[index]].iloc[:, 2].to_numpy()
            data_phase = dic_data[tags_array[index]].iloc[:, 3].to_numpy()

            for k in range(len(data_rssi)):
                if useful_rows_length <= k:
                    continue
                res_rssi[i][j][k] = data_rssi[k]
                res_phase[i][j][k] = data_phase[k]
                res_time[i][j][k] = data_time[k]
            index += 1
    print("rssi、phase提取完成")
    return res_rssi, res_phase, res_time
