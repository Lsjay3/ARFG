# Jay的开发时间：2022/8/2  21:37
import numpy as np
import matplotlib.pyplot as plt
print(np.ones((50,)) / 50)
# 均值滤波简单实现
mylist = [1, 2, 3, 4, 5, 6, 7]
N = 3
cumsum, moving_aves = [0], []
# for i, x in enumerate(mylist, 1):
#     cumsum.append(cumsum[i-1] + x)
#     if i>=N:
#         moving_ave = (cumsum[i] - cumsum[i-N])/N
#         #can do stuff with moving_ave here
#         moving_aves.append(moving_ave)
# print(moving_aves)
def np_move_avg(a, n, mode="same"):
    return (np.convolve(a, np.ones((n,)) / n, mode=mode))
print(np_move_avg(mylist,N,'full'))
# modes = ['full', 'same', 'valid']
# for m in modes:
#     plt.plot(np_move_avg(np.ones((200,)), 50, mode=m));
# plt.axis([-10, 251, -.1, 1.1]);
# plt.legend(modes, loc='lower center');
# plt.show()


