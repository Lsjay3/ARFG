# Jay的开发时间：2022/8/28  11:41
import numpy as np
import matplotlib.pyplot as plt
import GetData
from scipy.signal import savgol_filter

# Q为这一轮的心里的预估误差
Q = 0.00001
# R为下一轮的测量误差
R = 0.1
# Accumulated_Error为上一轮的估计误差，具体呈现为所有误差的累计
Accumulated_Error = 1
# 初始旧值
kalman_adc_old = 0
SCOPE = 50


def kalman(ADC_Value):
    global kalman_adc_old
    global Accumulated_Error
    # 新的值相比旧的值差太大时进行跟踪
    if abs(ADC_Value - kalman_adc_old) / SCOPE > 0.25:
        Old_Input = ADC_Value * 0.382 + kalman_adc_old * 0.618
    else:
        Old_Input = kalman_adc_old
    # 上一轮的 总误差=累计误差^2+预估误差^2
    Old_Error_All = (Accumulated_Error ** 2 + Q ** 2) ** (1 / 2)
    # R为这一轮的预估误差
    # H为利用均方差计算出来的双方的相信度
    H = Old_Error_All ** 2 / (Old_Error_All ** 2 + R ** 2)
    # 旧值 + 1.00001/(1.00001+0.1) * (新值-旧值)
    kalman_adc = Old_Input + H * (ADC_Value - Old_Input)
    # 计算新的累计误差
    Accumulated_Error = ((1 - H) * Old_Error_All ** 2) ** (1 / 2)
    # 新值变为旧值
    kalman_adc_old = kalman_adc
    return kalman_adc


sheet = Getdata('E:\笨比J\RFID\Impinj R420\实验数据\十二标签\十二标签数据.xlsx')
array = sheet.col_values(4)[1:]

# s = np.random.normal(0, 5, 200)
# test_array = array + s
# plt.plot(test_array)
adc = []
# for i in range(200):
#     adc.append(kalman(test_array[i]))
savgol_array = savgol_filter(array, 27, 2)
plt.plot(savgol_array)
plt.plot(array)
# plt.plot(savgol_array)
plt.show()
