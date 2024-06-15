import numpy as np
from scipy.io import loadmat


def OrderAllocate2(chro1, M, N):
    # 加载随机生成数据，准备时间和加工时间
    loaded_parameter = loadmat('parameter.mat')
    pt = loaded_parameter['pt']
    wt = loaded_parameter['wt']
    SSEc = loaded_parameter['SSEc']
    NEcpt = loaded_parameter['NEcpt']
    WEcpt = loaded_parameter['WEcpt']

    SSEc = np.squeeze(SSEc)
    NEcpt = np.squeeze(NEcpt)
    WEcpt = np.squeeze(WEcpt)

    # print("pt shape:", pt.shape)
    # print("wt shape:", wt.shape)
    # print("SSEc shape:", SSEc.shape)
    # print("NEcpt shape:", NEcpt.shape)
    # print("WEcpt shape:", WEcpt.shape)

    EnergyConsumption = np.zeros((M, N))  # 各机器加工能耗
    Mindex = np.zeros((M, N), dtype=int)  # 索引
    Mc = np.zeros((N, M))  # 初始化订单在每台机器上的总能耗矩阵
    SelectionProbabilityTemp1 = np.zeros((N, M))
    SelectionProbabilityTemp2 = np.zeros(M)  # 初始化选择概率临时矩阵
    SelectionProbability = np.zeros((N, M))  # 初始化选择概率矩阵
    total_sum = 0  # 总和
    k = np.ones(M, dtype=int)  # 计数器

    for i in range(N):
        # 计算订单在每台机器上的能耗
        for j in range(M):
            for p in range(k[j]):
                if p == 0:
                    EnergyConsumption[j, p] = WEcpt[j] * wt[chro1[i] - 1, j] # 确保是标量
                    Mc[chro1[i] - 1, j] = EnergyConsumption[j, k[j] - 1]
                else:
                    prev_index = Mindex[j, p - 1] - 1  # MATLAB 索引从 1 开始，Python 从 0 开始
                    if NEcpt[j] * pt[chro1[i] - 1, prev_index]>= SSEc[j]:  # 启停能耗大于空载能耗，选择关机
                        EnergyConsumption[j, p] = EnergyConsumption[j, p - 1] + SSEc[j] + WEcpt[j] * wt[
                            chro1[i] - 1, j]
                        Mc[chro1[i] - 1, j] = EnergyConsumption[j, k[j] - 1]
                    else:
                        EnergyConsumption[j, p] = EnergyConsumption[j, p - 1] + NEcpt[j] * pt[
                            chro1[i] - 1, prev_index] + WEcpt[j] * wt[chro1[i] - 1, j]
                        Mc[chro1[i] - 1, j] = EnergyConsumption[j, k[j] - 1]
            total_sum += Mc[chro1[i] - 1, j]

        # 计算订单选择每台机器的概率
        for l in range(M):
            SelectionProbabilityTemp1[chro1[i] - 1, l] = Mc[chro1[i] - 1, l] / total_sum  # 计算订单在每台机台上的选择概率

        SelectionProbabilityTemp2 = SelectionProbabilityTemp1[chro1[i] - 1, :]
        sorted_indices = np.argsort(SelectionProbabilityTemp2)
        for o in range(M):
            SelectionProbability[chro1[i] - 1, o] = SelectionProbabilityTemp2[sorted_indices[-o - 1]]

        # 选择总能耗最小的机器
        Cmax_min = np.max(SelectionProbability[chro1[i] - 1, :])  # 总能耗最小值
        temp = SelectionProbability[chro1[i] - 1, :]
        Machinenumber = np.where(temp == Cmax_min)[0]  # 找到满足条件的机器编号
        Machinenumber = np.random.choice(Machinenumber) # 随机选择一个机器

        # 订单进行选择
        Mindex[Machinenumber, k[Machinenumber] - 1] = chro1[i]  # 记录订单分配的机器位置
        k[Machinenumber] += 1  # 更新计数器
        chro2 = [0] * N
        chro2[i] = Machinenumber + 1  # Python 索引从 0 开始，需要加 1

        # 订单选择机器完成后进行多余参数设置
        # for g in range(M):
        #     if g != Machinenumber:
        #         EnergyConsumption[g, k[g] - 1] = 0
        #
        # total_sum = 0

    return chro2