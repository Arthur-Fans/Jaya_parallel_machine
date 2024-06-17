import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
# from Jaya_Parallel_Machine.Jaya import non_dominated_sorting, calc_crowding_distance, sort_population
from orderAllocateByTime import OrderAllocate1
from orderAllocateByEnergy import OrderAllocate2
from fitness import Fitness

# 初始化种群
N = 5  # 订单数量
M = 3  # 机器数量
POP_SIZE = 12  # 种群大小
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


# 定义 Individual 类
class Individual:
    def __init__(self):
        self.Position1 = []  # 订单加工序列
        self.Position2 = []  # 机器加工序列
        self.Cost = []
        self.Information = {
            'M_index': [],
            'Finish_time': [],
            'Cmax': [],
            'Energy_consumption': []
        }
        self.Rank = []
        self.DominationSet = []
        self.DominatedCount = []
        self.CrowdingDistance = []


# 创建空的种群列表
pop = [Individual() for _ in range(POP_SIZE)]


# 定义需要的函数
def randperm(n):
    return random.sample(range(n), n)


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
    #
    M_index, Finish_time, Cmax, Final_energy_consumption = Fitness(pop[i].Position1, pop[i].Position2, M, N)
    # 最大完成时间和总能耗数组
    pop[i].Cost = [Cmax, Final_energy_consumption]
    # 种群信息
    pop[i].Information['M_index'] = M_index
    pop[i].Information['Finish_time'] = Finish_time
    pop[i].Information['Cmax'] = Cmax
    pop[i].Information['Energy_consumption'] = Final_energy_consumption


# #  非支配排序
# pop, F = non_dominated_sorting(pop)
# #
# #  计算拥挤度
# pop = calc_crowding_distance(pop, F)
# #
# #  种群个体排序
# pop, F = sort_population(pop)

# 获取最佳个体的信息
# M_index = np.array(M_index) - 1  # 将订单编号从 1 调整为 0
# Finish_time = np.array(Finish_time)
# #
# # # 确保 M_index 和 Finish_time 的维度正确
# M, N = M_index.shape

# 绘制甘特图
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
    # print("Finish_time:")
    # print(Finish_time)
    # print("Cmax:", Cmax)
    # print("Final_energy_consumption:", Final_energy_consumption)
    print("Preparation Time Matrix (pt):\n", pt)
    print("Processing Time Matrix (t):\n", wt)
    for i in range(len(pop)):
        print(pop[i].Position1, pop[i].Position2,pop[i].Cost,pop[i].Information['Finish_time'])

    # print(pop[i].Cost)
    # print(pop[i].Information)

    # print("Start/Stop Energy Consumption (Sec):\n", SSEc)
    # print("No-load Energy Consumption Coefficient (NEcpt):\n", NEcpt)
    # print("Energy Consumption Coefficient (WEcpt):\n", WEcpt)
