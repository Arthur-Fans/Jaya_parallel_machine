import numpy as np
from scipy.io import loadmat

def Fitness(chro1, chro2, M, N):
    # 加载数据
    loaded_parameter = loadmat('parameter.mat')
    pt = loaded_parameter['pt']  # 准备时间
    wt = loaded_parameter['wt']  # 加工时间
    SSEc = loaded_parameter['SSEc']  # 启停能耗
    NEcpt = loaded_parameter['NEcpt']
    WEcpt = loaded_parameter['WEcpt']

    SSEc = np.squeeze(SSEc)
    NEcpt = np.squeeze(NEcpt)
    WEcpt = np.squeeze(WEcpt)

    Finish_time = np.zeros((M, N))  # 各机器加工完成时间
    Energy_consumption = np.zeros((M, N))  # 各机器加工能耗
    Final_energy_consumption = 0  # 总能耗
    M_index = np.zeros((M, N), dtype=int)  # 索引
    k = np.ones(M, dtype=int)

    # 解码，获取机器加工信息
    for i in range(N):
        for j in range(M):
            if chro2[i] == j + 1:  # MATLAB 索引从 1 开始，Python 从 0 开始
                M_index[j, k[j] - 1] = chro1[i]
                k[j] += 1

    # 求最大完工时间及总能耗
    # 求最大完工时间
    count = np.sum(M_index != 0, axis=1)
    for j in range(M):
        for k in range(count[j]):
            if k == 1:
                Finish_time[j, k] = pt[M_index[j, k] - 1, j]
            else:
                Finish_time[j, k] = Finish_time[j, k - 1] + pt[M_index[j, k] - 1, M_index[j, k - 1] - 1] + wt[M_index[j, k] - 1, j]

    Cmax = np.max(Finish_time)

    # 求总能耗
    for j in range(M):
        for k in range(count[j]):
            if k == 0:
                Energy_consumption[j, k] = WEcpt[j] * pt[M_index[j, k] - 1, j]
            else:
                if NEcpt[j] * pt[M_index[j, k] - 1, M_index[j, k - 1] - 1] >= SSEc[j]:
                    Energy_consumption[j, k] = Energy_consumption[j, k - 1] + SSEc[j] + WEcpt[j] * pt[M_index[j, k] - 1, j]
                else:
                    Energy_consumption[j, k] = Energy_consumption[j, k - 1] + NEcpt[j] * pt[M_index[j, k] - 1, M_index[j, k - 1] - 1] + WEcpt[j] * pt[M_index[j, k] - 1, j]

    M_Energy_consumption = np.max(Energy_consumption, axis=1)

    for i in range(M):
        if i == 0:
            Final_energy_consumption = M_Energy_consumption[i]
        else:
            Final_energy_consumption += M_Energy_consumption[i]

    return M_index, Finish_time, Cmax, Final_energy_consumption
