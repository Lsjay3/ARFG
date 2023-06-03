# Jay的开发时间：2022/8/3  10:17
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import PhaseUnwrap

def val_eb(x, y,gap):
 # 转折点前后数据的离散程度相差很大，用这个离散差异拟合一条曲线，可能会有奇效。
 # 对每一个数据点（离散数据点拟合的曲线），向前向后分别取一段等长区间，分别求方差，用方差表示离散程度。用向前区间的方差/向后区间的方差，表示离散差异。
 # 每个数据点求得的离散差异形成的曲线如下，极值点x=790，正是转折点。
    col = []
    y_col = []
    for i in range(gap, len(y) - gap):
        col.append(np.var(y[i - gap:i]) / np.var(y[i:i + gap]))                # / np.var(y[i:i + gap])
        y_col.append(i)
    return col, y_col
    pass
def pl():
    n = 1000
    x = np.linspace(1, 10, n)
    # 绘制曲线
    noise = np.random.normal(0, 0.008, 1000)
    # print(noise)
    y = np.sin(x)
    plt.plot(x, y, color="red", linewidth='0.1')
    # x_smooth = savgol_filter(y, 101, 3)
    y = savgol_filter(y, 201, 3)
    # 方差求拐点
    v, y_v = val_eb(x, y, 100)
    vsorted = sorted(v, reverse=True)
    for i in range(len(vsorted)):
        print(x[y_v[v.index(vsorted[i])]])
    # print(vsorted)
    i = y_v[v.index(max(v))]
    # print("方差:", max(v))
    print("the ebow is ", x[i], y[i])
    # # print(x[i], y[i], x[i + 1], y[i + 1], x[i - 1], y[i - 1])
    plt.plot(x, y, color="green", linewidth='0.1')
    plt.show()
pl()