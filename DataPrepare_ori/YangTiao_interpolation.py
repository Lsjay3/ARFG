# Jay的开发时间：2022/9/20  15:14
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 导入数据

# 数据准备
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 定义数据点
Y = [3700, 3900, 3200, 3750,
     3600,
     4000,
     3905,
     3750,
     3600,
     6300
     ]  # 定义数据点
x = np.arange(0, len(Y), 0.15)  # 定义观测点

# 进行样条差值
import scipy.interpolate as spi

# 进行一阶样条差值
ipo1 = spi.splrep(X, Y, k=1)  # 源数据点导入，生成参数
iy1 = spi.splev(x, ipo1)  # 根据观测点和样条参数，生成插值

# 进行三次样条拟合
ipo3 = spi.splrep(X, Y, k=3)  # 源数据点导入，生成参数
iy3 = spi.splev(x, ipo3)  # 根据观测点和样条参数，生成插值

##作图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
ax1.plot(X, Y, label='沪深300')
ax1.plot(x, iy1, 'r.', label='插值点')
ax1.set_ylim(min(Y) - 10, max(Y) + 10)
ax1.set_ylabel('指数')
ax1.set_title('线性插值')
ax1.legend()
ax2.plot(X, Y, label='沪深300')
ax2.plot(x, iy3, 'b.', label='插值点')
ax2.set_ylim(min(Y) - 10, max(Y) + 10)
ax2.set_ylabel('指数')
ax2.set_title('三次样条插值')
ax2.legend()
plt.show()
