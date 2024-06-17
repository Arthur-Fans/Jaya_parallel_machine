import numpy as np
from scipy.io import loadmat


def OrderAllocate1(chro1, M, N):
    # 加载随机生成数据，准备时间和加工时间
    loaded_parameter = loadmat('parameter.mat')
    pt = loaded_parameter['pt']
    wt = loaded_parameter['wt']

    FinishTime = np.zeros((M, N))  # 各机器加工完成时间
    Mindex = np.zeros((M, N))  # 索引
    Tc = np.zeros((N, M))  # 初始化订单在每台机器上的完工时间矩阵
    SelectionProbabilityTemp1 = np.zeros((N, M))  # 初始化选择概率临时矩阵
    SelectionProbabilityTemp2 = np.zeros((1, M))
    SelectionProbability = np.zeros((N, M))  # 初始化选择概率矩阵
    chro2 = np.zeros(N, dtype=int)  # 初始化订单选择的机器编号
    total_sum = 0  # 总和
    k = np.ones(M, dtype=int)  # 计数器 k[j]=2表示第j台机器上的第2个订单

    for i in range(N):
        # 计算订单在每台机器上的完工时间
        for j in range(M):
            if k[j] == 1:  # 是否是第一个
                FinishTime[j, k[j] - 1] = wt[chro1[i], j]
                Tc[chro1[i], j] = FinishTime[j, k[j] - 1]
            else:
                FinishTime[j, k[j] - 1] = FinishTime[j, k[j] - 2] + pt[chro1[i], int(Mindex[j, k[j] - 2])] + wt[
                    chro1[i], j]  # 上一个完成时间+当前订单在前一个订单机器上的准备时间+当前订单加工时间
                Tc[chro1[i], j] = FinishTime[j, k[j] - 1]
            total_sum += Tc[chro1[i], j]
        # print('Tc[chro1[i], j]',Tc[chro1[i], j])
        # 计算订单选择每台机器的概率
        for l in range(M):
            SelectionProbabilityTemp1[chro1[i], l] = Tc[chro1[i], l] / total_sum  # 计算订单在每台机台上的选择概率

        # 获取当前订单chro[i]所有机器的选择概率
        SelectionProbabilityTemp2 = SelectionProbabilityTemp1[chro1[i], :]
        # # 获取从小到大的索引值
        # sorted_indices = np.argsort(SelectionProbabilityTemp2)
        # # 创建一个新数组来存储排序后的序号
        # rank_array = np.empty_like(sorted_indices)
        # # 将排序后的序号赋值给rank_array
        # rank_array[sorted_indices] = np.arange(len(SelectionProbabilityTemp2))
        #
        # # 找对应机器
        # for o in range(M):
        #     SelectionProbability[chro1[i] - 1, o] = SelectionProbabilityTemp2[rank_array[M - 1 - o]]
        # 对选择概率进行排序
        Y = np.sort(SelectionProbabilityTemp2)
        I = np.argsort(SelectionProbabilityTemp2)

        # 调整选择概率
        for o in range(M):
            idx = np.where(I == o)[0][0]
            SelectionProbability[chro1[i] - 1, o] = Y[M - 1 - idx]

        # 选择最大完工时间最小的机器
        Cmax_min = np.max(SelectionProbability[chro1[i], :])  # 最大完工时间最小值
        temp = SelectionProbability[chro1[i], :]
        Machinenumber = np.where(temp == Cmax_min)[0]  # 找到满足条件的机器编号
        # print('Machinenumber', Machinenumber)
        Machinenumber = np.random.choice(Machinenumber)  # 随机选择一个机器

        # 订单进行选择
        Mindex[Machinenumber, k[Machinenumber] - 1] = chro1[i]  # 记录订单分配的机器位置
        k[Machinenumber] += 1  # 更新计数器
        chro2[i] = Machinenumber

        # 订单选择机器完成后进行多余参数设置
        for g in range(M):
            if g != Machinenumber:
                FinishTime[g, k[g] - 1] = 0

        total_sum = 0

    return chro2.tolist()
