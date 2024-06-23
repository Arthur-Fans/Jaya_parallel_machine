import random
from datetime import time, datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from drawing import CalculatedEstimate, PlotCosts
from individual import Individual
from Jaya import non_dominated_sorting, calc_crowding_distance, sort_population, populationEvolve
from orderAllocateByTime import OrderAllocate1
from orderAllocateByEnergy import OrderAllocate2
from fitness import Fitness

# 初始化种群
N = 40  # 订单数量
M = 10  # 机器数量
POP_SIZE = 100  # 种群大小
MAX_GEN = 100  # 最大迭代次数
N1 = POP_SIZE / 4  # 以时间初始化个体数
N2 = POP_SIZE / 2  # 以能耗初始化个体数
N3 = POP_SIZE / 4  # 随机初始化初始化个体数

loaded_parameter = loadmat('parameter.mat')
pt = loaded_parameter['pt']
wt = loaded_parameter['wt']
SSEc = loaded_parameter['SSEc']
NEcpt = loaded_parameter['NEcpt']
WEcpt = loaded_parameter['WEcpt']

# 创建空的种群列表
pop = [Individual() for _ in range(POP_SIZE)]


# 定义需要的函数
def randperm(n):
    return random.sample(range(1, n + 1), n)


# 记录开始时间
# start_time = datetime.now()

# 初始化种群
for i in range(POP_SIZE):
    # 订单编码顺序
    pop[i].Position1 = randperm(N)
    # 机器编码顺序
    if i < N1:
        pop[i].Position2 = OrderAllocate1(pop[i].Position1, M, N)
    elif N1 <= i < N1 + N2:
        pop[i].Position2 = OrderAllocate2(pop[i].Position1, M, N)
    else:
        pop[i].Position2 = np.random.randint(0, M, size=N).tolist()

    M_index, Finish_time, Cmax, Final_energy_consumption = Fitness(pop[i].Position1, pop[i].Position2, M, N)
    # 最大完成时间和总能耗数组
    pop[i].Cost = [Cmax, Final_energy_consumption]
    # 种群信息
    pop[i].Information['M_index'] = M_index
    pop[i].Information['Finish_time'] = Finish_time
    pop[i].Information['Cmax'] = Cmax
    pop[i].Information['Energy_consumption'] = Final_energy_consumption

# 非支配排序
pop, F = non_dominated_sorting(pop)
# 计算拥挤度
pop = calc_crowding_distance(pop, F)
# 种群个体排序
pop, F = sort_population(pop)

D1 = []  # 存储最大完工时间
D2 = []  # 存储平均总能耗
D3 = []  # 存储帕累托解个数
i = 1
# Jaya 算法流程
for i in range(MAX_GEN):
    # 非支配排序
    pop, F = non_dominated_sorting(pop)
    # 计算拥挤度
    pop = calc_crowding_distance(pop, F)
    # 种群个体排序
    pop, F = sort_population(pop)
    # 获取种群中最优和最差的个体
    best = pop[0]
    worst = pop[-1]
    # 种群进化
    pop = populationEvolve(pop, best, worst, POP_SIZE)
    # 非支配排序
    pop, F = non_dominated_sorting(pop)
    # 计算拥挤度
    pop = calc_crowding_distance(pop, F)
    # 种群个体排序
    pop, F = sort_population(pop)
    # # 记录帕累托第一层的评估值
    EvaluationValue = CalculatedEstimate(pop, len(F[0]))
    D1.append(EvaluationValue[0])  # 完成时间
    D2.append(EvaluationValue[1])  # 能耗
    D3.append(len(F[0]))
#     # 提取第一个非支配前沿的个体
#     F1 = [pop[i] for i in F[0]]
#     # 结果显示
#     print(f'Iteration {i + 1}: Number of F1 Members = {len(F1)}')
#
#     # 动态画图
#     plt.figure(1)
#     PlotCosts(F1)
#     plt.pause(0.01)
#
    # 计算运行时间，超过一定时间则停止
    # end_time = datetime.now()
    # elapsed_time = end_time - start_time
    # if elapsed_time.total_seconds() > M * N:
    #     break

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 记录帕累托第一层的评估值
EvaluationValue = CalculatedEstimate(pop, len(F[0]))
D1.append(EvaluationValue[0])  # 完成时间
D2.append(EvaluationValue[1])  # 能耗
D3.append(len(F[0]))


# 提取第一个非支配前沿的个体，绘制帕累托前沿（第一层）
F1 = [pop[i] for i in F[0]]
plt.figure(1)
PlotCosts(F1)

# 绘制迭代优化效果图
x = range(1, MAX_GEN + 2)
plt.figure(2)
plt.subplot(3, 1, 1)
plt.plot(x, D1, '-*r')
plt.xlabel('迭代次数')
plt.ylabel('平均最大完工时间')

plt.subplot(3, 1, 2)
plt.plot(x, D2, '-og')
plt.xlabel('迭代次数')
plt.ylabel('平均总能耗')

# plt.subplot(3, 1, 3)
# plt.plot(x, D3, '-xb')
# plt.xlabel('迭代次数')
# plt.ylabel('帕累托解集个数')

plt.show()

# M_index = np.array(M_index) - 1  # 将订单编号从 1 调整为 0
# Finish_time = np.array(Finish_time)
# #
# # # 确保 M_index 和 Finish_time 的维度正确
# M, N = M_index.shape

def plot_gantt(M_index, Finish_time):
    colors = plt.cm.tab20(np.linspace(0, 1, N))  # 获取颜色映射

    plt.figure(figsize=(12, 6))

    for i in range(M):
        start_time = 0  # 初始化开始时间
        first_order = True  # 标记是否是第一个订单
        for j in range(N):
            job_id = int(M_index[i, j])
            if job_id > 0:  # 确保订单编号有效
                job_id -= 1  # 调整为0索引
                wt_time = pt[job_id, i]  # 加工时间
                pt_time = 0 if first_order else pt[job_id, i]  # 第一个订单无准备时间

                # 画准备时间（蓝色），如果不是第一个订单
                if not first_order:
                    plt.barh(i, pt_time, left=start_time, height=0.5, color='blue', edgecolor='black')
                    start_time += pt_time  # 更新开始时间

                # 画加工时间（绿色）
                plt.barh(i, wt_time, left=start_time, height=0.5, color='green', edgecolor='black')
                plt.text(start_time + wt_time / 2, i, f'Job {job_id + 1}', ha='center', va='center', color='white')
                start_time += wt_time  # 更新开始时间

                first_order = False  # 第一个订单处理完毕

    plt.xlabel('Time')
    plt.ylabel('Machine')
    plt.title('Gantt Chart')
    plt.yticks(range(M), [f'Machine {i + 1}' for i in range(M)])
    plt.xticks(np.arange(0, np.max(Finish_time) + 1, step=max(1, int(np.max(Finish_time) / 20))))  # 根据最大完成时间设置 x 轴刻度
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_gantt(M_index, Finish_time)
    # print(M_index)
    # print("Finish_time:"+ Finish_time)
    # print("Cmax:"+ Cmax)
    print(F)
    # print(D1)
    # print(D2)
    # for i in range(len(pop)):
    #     # print(pop[i].Position1, pop[i].Position2,pop[i].Cost,pop[i].Information['Finish_time'],pop[i].Information['Cmax'])
    #     print(pop[i].Cost)
