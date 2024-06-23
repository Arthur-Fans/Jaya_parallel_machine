import matplotlib.pyplot as plt
from individual import Individual


# 计算评估值
def CalculatedEstimate(pop, nF1):
    cost1 = 0
    cost2 = 0
    EvaluationValue = []
    # 累加前 nF1 个体的成本
    for i in range(nF1):
        cost1 += pop[i].Cost[0]
        cost2 += pop[i].Cost[1]

    # 计算平均成本
    EvaluationValue.append(cost1 / nF1)
    EvaluationValue.append(cost2 / nF1)

    return EvaluationValue

# 绘制最终结果
def PlotCosts(pop):
    # 提取成本
    # costs = [Individual.Cost for Individual in pop]
    # 将成本列表展平成一个单独的列表
    flat_costs = [cost for Individual in pop for cost in Individual.Cost]

    # 获取成本的数量
    n = len(flat_costs)

    # 初始化 x 和 y 坐标
    x = [0] * (n // 2)
    y = [0] * (n // 2)

    # 填充 x 和 y 坐标
    for i in range(n // 2):
        x[i] = flat_costs[2 * i]
        y[i] = flat_costs[2 * i + 1]

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制散点图
    plt.plot(x, y, 'r*', markersize=8)

    # 添加坐标标签
    plt.xlabel('最大完工时间（Cmax）')
    plt.ylabel('总能耗（Energy consumption）')
    # 添加标题
    plt.title('Pareto解集')
    # 显示网格
    plt.grid(True)
    # 显示图表
    plt.show()
