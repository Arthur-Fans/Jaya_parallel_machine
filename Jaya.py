import numpy as np

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
                d[sorted_indices[i], j] = abs(costs[sorted_indices[i+1], j] - costs[sorted_indices[i-1], j]) / abs(costs[sorted_indices[-1], j] - costs[sorted_indices[0], j])

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
