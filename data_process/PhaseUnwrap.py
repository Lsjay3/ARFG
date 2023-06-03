# Jay的开发时间：2022/8/1  15:03
import xlrd
import numpy as np
import xlwt

# 平均滤波
def np_move_avg(a, n, mode="same"):
    return (np.convolve(a, np.ones((n,)) / n, mode=mode))

def getPhase():
    data = xlrd.open_workbook('单标签RSSI与相位.xlsx')
    # 选择页数为第1页
    sheet1 = data.sheets()[0]
    # 数据总行数
    nrows = sheet1.nrows
    print('数据总行数：', nrows)
    phase = sheet1.col_values(3)[1:]
    return phase

# # 获取表中第三行的数据
# x = sheet1.row_values(2)
# print('第3行: ', x)
#
# # 获取表中第二列的数据
# y = sheet1.col_values(1)
# print('第二列： ', y)
# # 获取表中第二列且不要第一个值的数据

# phase = np.array(phase)
# print('初始相位： ', phase)
# np.unwrap(phase)
# print('解包后相位： ', np.unwrap(phase))
# a = np.array([0.104310693576223, 0.098174770424681, 0.110446616727766, 0.0859029241215959, 0.0245436926061702])
# print('初始相位： ', a)
# np.unwrap(a)
# print('解包后相位： ', np.unwrap(a))
book = xlwt.Workbook(encoding='utf-8',style_compression=0)
def create_book():
    sheet = book.add_sheet('相位',cell_overwrite_ok=True)
    col = ('解包前','解包后','full', 'same', 'valid')
    for i in range(0,len(col)):
        sheet.write(0,i,col[i])
    return sheet

def get_phase_unwrap():
    phase_unwrap = np.unwrap(getPhase())
    return phase_unwrap

def write_phase():
    sheet = create_book()
    phase = getPhase()
    phase_unwrap = get_phase_unwrap()
    modes = ['full', 'same', 'valid']
    N = 12                                              #滑动窗口大小
    for i in range(1, len(phase) - N):
        sheet.write(i, 0, phase[i - 1])
        sheet.write(i, 1, phase_unwrap[i - 1])
        for j in range(3):
            # 将滤波结果写入文件
            phase_after_filter = np_move_avg(phase_unwrap, N, modes[j])
            sheet.write(i, 2 + j, phase_after_filter[i - 1])

# print(pow(7.079457843841373e-08,1/2))


# write_phase()
#
# savepath = 'E:\笨比J\RFID\Code\DataPrepare\相位.xls'
# book.save(savepath)


