# Jay的开发时间：2022/8/29  9:53
import pandas as pd
import xlrd
import numpy as np
import xlwt
from xlutils.copy import copy
import time
import datetime
import matplotlib.pyplot as plt
import csv
import cv2  # cv2 即opencv的库

# 数据路径，标签个数， 静止 0/手势 1， 存储路径
def Part_tags(path, tag_num, flag, path_store):
    data = pd.read_csv(path, usecols=['// Timestamp',  ' EPC', ' RSSI', ' PhaseAngle'])
    # print(data.iloc[1, 1])
    if isinstance(data.iloc[1, 0], str):
        time_start = datetime.datetime.strptime(data.iloc[0, 0][:-7], "%Y-%m-%dT%H:%M:%S.%f")

        start_stamp = int(time.mktime(time_start.timetuple()) * 1000000.0 + time_start.microsecond)
        max_timestamp = 0
        for i in range(len(data)):
            new_t = datetime.datetime.strptime(data.iloc[i, 0][:-7], "%Y-%m-%dT%H:%M:%S.%f")
            obj_stamp = int(time.mktime(new_t.timetuple()) * 1000000.0 + new_t.microsecond)
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
    # if flag == 1:
    #     print(data)
    Tags_name = list()
    Tags = list()
    book = xlrd.open_workbook(path_store)
    book_prepare = copy(book)
    col = ["静止时间戳", "RSSI_静止", "Phase_静止", "手势时间戳", "RSSI_手势", "Phase_手势"]
    # print(data[' EPC'])

    i = 0
    # 获得标签名字
    while len(Tags_name) < tag_num:
        if data.iloc[i, 1] not in Tags_name:
            Tags_name.append(str(data.iloc[i, 1]))
        i += 1
    # print(Tags_name)
    for names in Tags_name:
        # 筛选不同标签数据，并和其EPC一一对应
        # print(data.loc[data[' EPC'] == int(names), :])
        Tags.append(data.loc[data[' EPC'] == names, :])
        # print(data.loc[data[' EPC'] == names, :])
    dic_data = dict(zip(Tags_name, Tags))
    # print(help(book_prepare))
    # 写入工作表
    # for names in Tags_name:
    #     if names in book.sheet_names():
    #         sheet = book_prepare.get_sheet(names)
    #     else:
    #         sheet = book_prepare.add_sheet(names, cell_overwrite_ok=True) # 创建一张表单
    #     timestamp = dic_data[names].iloc[:, 0].to_numpy()
    #     # print(timestamp)
    #     rssi = dic_data[names].iloc[:, 2].to_numpy()
    #     phase = dic_data[names].iloc[:, 3].to_numpy()
    #     for j in range(len(col)):
    #         sheet.write(0, j, col[j])
    #     for j in range(1, len(rssi) + 1):
    #         if flag == 0:
    #             sheet.write(j, 0, str(timestamp[j - 1]))
    #             sheet.write(j, 1, str(rssi[j - 1]))
    #             sheet.write(j, 2, str(phase[j - 1]))
    #         else:
    #             # print("rssi:", rssi[j - 1])
    #             sheet.write(j, 3, str(timestamp[j - 1]))
    #             sheet.write(j, 4, str(rssi[j - 1]))
    #             sheet.write(j, 5, str(phase[j - 1]))
    # book_prepare.save(path_store)
    # if flag == 0:
    #     print("标签分离(静止状态)已完成")
    # else:
    #     print("标签分离(手势状态)已完成")
    return path_store, dic_data, max_timestamp

# 将所有标签数据写入一个任务表sheet中
def Together_tags(path, dic_data, flag, col):
    book = xlrd.open_workbook(path)
    book_prepare = copy(book)
    if flag == 0:
        sheet_name = ['RSSI_静止', 'Phase_静止']
    else:
        sheet_name = ['RSSI_手势', 'Phase_手势']
    for sheetnames in sheet_name:
        if sheetnames in book.sheet_names():
            sheet = book_prepare.get_sheet(sheetnames)
        else:
            sheet = book_prepare.add_sheet(sheetnames, cell_overwrite_ok=True)  # 创建一张表单
        for j in range(len(col)):
            sheet.write(0, j, col[j])
            if sheetnames == 'RSSI_静止' or sheetnames == 'RSSI_手势':
                # print(dic_data[col[j]])
                data = dic_data[col[j]].iloc[:, 2].to_numpy()
            elif sheetnames == 'Phase_静止' or sheetnames == 'Phase_手势':
                data = np.unwrap(dic_data[col[j]].iloc[:, 3].to_numpy())
            for i in range(1, len(data) + 1):
                sheet.write(i, j, str(data[i - 1]))
    if flag == 0:
        print("原始数据写入(静止状态)已完成")
    else:
        print("原始数据写入(手势状态)已完成")
    book_prepare.save(path)

# 将原始标签数据转化为numpy数组形式（height * width * num）
def Transfer_np(dic_data, width, height, col):
    l = len(dic_data[col[0]])
    for i in range(width * height):
        if len(dic_data[col[i]]) < l:
                l = len(dic_data[col[i]])
    # print(l)
    res_rssi = np.zeros((height, width, l), dtype=np.float64)
    res_phase = np.zeros((height, width, l), dtype=np.float64)
    res_time = np.zeros((height, width, l), dtype=np.float64)
    index = 0
    for i in range(height):
        for j in range(width):
            data_time = dic_data[col[index]].iloc[:, 0].to_numpy()
            data_rssi = dic_data[col[index]].iloc[:, 2].to_numpy()
            data_phase = dic_data[col[index]].iloc[:, 3].to_numpy()

            for k in range(len(data_rssi)):
                if l <= k:
                    continue
                res_rssi[i][j][k] = data_rssi[k]
                res_phase[i][j][k] = data_phase[k]
                res_time[i][j][k] = data_time[k]
            index += 1
    print("rssi、phase提取完成")
    return res_rssi, res_phase, res_time