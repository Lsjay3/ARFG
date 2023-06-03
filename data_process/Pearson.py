# Jay的开发时间：2022/8/25  9:31
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import xlrd
from xlutils.copy import copy
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F


data = xlrd.open_workbook('E:\笨比J\RFID\Impinj R420\实验数据\十二标签\十二标签理论信号.xlsx')
book_prepare = copy(data)
sheet = data.sheets()[1]
sheet_copy = book_prepare.get_sheet(1)
actual_P = sheet.col_values(18)
print(actual_P)
pearson = list()
for i in range(12):
    theore_P = sheet.col_values(i)
    pc = pearsonr(theore_P,actual_P)
    pearson.append(pc[0])
    print("相关系数" + str(i + 1) + "：", pc[0])
pearson = torch.tensor(pearson)
print(pearson)
I_x_y = F.softmax(pearson, dim= 0)
# scaler = MinMaxScaler(feature_range=(0, 1))  # 将数据归一到0到1，可以根据数据特点归一到-1到1
# I_x_y = scaler.fit_transform(pearson)  # 归一化
print("似然估计：",I_x_y)


#