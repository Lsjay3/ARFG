# Jay的开发时间：2022/8/27  11:06
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import xlwt
from xlutils.copy import copy

data = xlrd.open_workbook('E:\笨比J\RFID\Impinj R420\实验数据\十二标签\十二标签数据.xlsx')
book_prepare = copy(data)

sheet = data.sheets()[0]                    # 负责将数据（phase）读出
sheet_copy = book_prepare.get_sheet(0)      # 负责将处理后的phase写入目标表
len = len(sheet.col_values(1)[1:])
# 假设一个时间片被分成了3个时隙，标签在这个三个时隙中随机挑选一个时隙进行响应，标签响应时隙不同导致数据间隔不一致
# 因此插值就是要将三个时隙数据全部填满，使得数据时间间隔一致
x = np.linspace(0, len * 3 - 1, len)

y = sheet.col_values(1)[1:]

xvals = np.linspace(0, len * 3 - 1, len * 3)

# 线性插值
yinterp = np.interp(xvals, x, y)

print(yinterp)
plt.plot(xvals, yinterp, 'x')
plt.show()


