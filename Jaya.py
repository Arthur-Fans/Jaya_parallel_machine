import copy

import numpy as np

from fitness import Fitness
from individual import Individual

N = 5  # 订单数量
M = 3  # 机器数量


# 支配关系判断
def dominates(x, y):
    """
    判断个体 x 是否支配个体 y
    """
    x = x.Cost
    y = y.Cost
    return all(x_i <= y_i for x_i, y_i in zip(x, y)) and any(x_i < y_i for x_i, y_i in zip(x, y))


# 非支配排序
def non_dominated_sorting(pop):
    nPop = len(pop)
    for i in range(nPop):
        pop[i].DominationSet = []
        pop[i].DominatedCount = 0

    F = [[]]
    for i in range(nPop):
        for j in range(i + 1, nPop):
            p = pop[i]
            q = pop[j]
            if dominates(p, q):
                p.DominationSet.append(j)
                q.DominatedCount += 1
            if dominates(q, p):
                q.DominationSet.append(i)
                p.DominatedCount += 1
            pop[i] = p
            pop[j] = q
        if pop[i].DominatedCount == 0:
            F[0].append(i)
            pop[i].Rank = 1

    k = 0
    while True:
        Q = []
        for i in F[k]:
            p = pop[i]
            for j in p.DominationSet:
                q = pop[j]
                q.DominatedCount -= 1
                if q.DominatedCount == 0:
                    Q.append(j)
                    q.Rank = k + 1
                pop[j] = q
        if not Q:
            break
        F.append(Q)
        k += 1

    return pop, F


# 计算拥挤度
def calc_crowding_distance(pop, F):
    for front in F:
        nobj = len(pop[front[0]].Cost)
        n = len(front)
        d = np.zeros((n, nobj))
        costs = np.array([pop[i].Cost for i in front])

        for j in range(nobj):
            sorted_indices = np.argsort(costs[:, j])
            d[sorted_indices[0], j] = np.inf
            d[sorted_indices[-1], j] = np.inf
            for i in range(1, n - 1):
                d[sorted_indices[i], j] = abs(costs[sorted_indices[i + 1], j] - costs[sorted_indices[i - 1], j]) / abs(
                    costs[sorted_indices[-1], j] - costs[sorted_indices[0], j])

        for i in range(n):
            pop[front[i]].CrowdingDistance = np.sum(d[i, :])

    return pop


# 个体排序，返回排序号的种群和非支配层次
def sort_population(pop):
    # 基于拥挤度排序
    pop.sort(key=lambda x: x.CrowdingDistance, reverse=True)
    # 基于排名排序
    pop.sort(key=lambda x: x.Rank)

    # 分组到非支配层次
    ranks = [ind.Rank for ind in pop]
    max_rank = max(ranks)
    F = [[] for _ in range(max_rank)]
    for i, rank in enumerate(ranks):
        F[rank - 1].append(i)

    return pop, F


# 种群进化
def populationEvolve(pop, best, worst, pop_size):
    # 创新新的初始种群
    newPop = [Individual() for _ in range(pop_size)]
    newPop[0] = best
    for i in range(1, pop_size):
        oldIndividual = pop[i]
        # 将当前个体进行进化操作
        newIndividual = individualEvolve(oldIndividual, best, worst)
        # 计算进化后个体的适应度值
        M_index, Finish_time, Cmax, Final_energy_consumption = Fitness(newIndividual.Position1,
                                                                       newIndividual.Position2, M, N)
        newIndividual.Cost = [Cmax, Final_energy_consumption]
        # 种群信息
        newIndividual.Information['M_index'] = M_index
        newIndividual.Information['Finish_time'] = Finish_time
        newIndividual.Information['Cmax'] = Cmax
        newIndividual.Information['Energy_consumption'] = Final_energy_consumption
        # 进化后优于进化前则替换为进化后的个体
        if dominates(newIndividual, oldIndividual):
            newPop[i] = newIndividual
        else:
            newPop[i] = oldIndividual

    return newPop


# 个体进化
def individualEvolve(oldIndividual, best, worst):
    newIndividual = copy.deepcopy(oldIndividual)
    # 订单编码部分
    index = []  # 记录要替换的位置
    value = []  # 记录被替换的值
    for i in range(len(oldIndividual.Position1)):
        if oldIndividual.Position1[i] == worst.Position1[i]:
            index.append(i)
            value.append(oldIndividual.Position1[i])

    for i in range(len(index)):
        flag = index[i]
        for j in range(len(best.Position1)):
            if best.Position1[j] in value:
                newIndividual.Position1[flag] = best.Position1[j]
                value.remove(best.Position1[j])
                break

    # 机器编码部分
    for i in range(len(oldIndividual.Position2)):
        # 如果当前个体和最劣个体机器选择编码部分编码值相同，则将最优个体相同码位上的编码复制到该码位中
        if oldIndividual.Position2[i] == worst.Position2[i]:
            newIndividual.Position2[i] = best.Position2[i]

    return newIndividual
