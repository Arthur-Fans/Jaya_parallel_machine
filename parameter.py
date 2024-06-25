import numpy as np
from scipy.io import savemat, loadmat

# 初始化变量
N = 5  # 订单数量
M = 3  # 机器数量

# 生成准备时间矩阵
pt = np.round(5 + (15 - 5) * np.random.rand(N, N))
# 生成加工时间矩阵
wt = np.round(20 + (50 - 20) * np.random.rand(N, M))
# 生成M台机器的启停能耗数组
SSEc = np.round(50 + (150 - 50) * np.random.rand(1, M))
# 生成M台机器的空载能耗系数数组
NEcpt = np.round(5 + (15 - 5) * np.random.rand(1, M))
# 生成M台机器的加工能耗系数数组
WEcpt = np.round(10 + (20 - 10) * np.random.rand(1, M))

# 将准备时间矩阵的对角元素设置为零，表示同一个订单接着自己，不需要准备时间
np.fill_diagonal(pt, 0)

# # 保存所有生成的矩阵和数组到一个.mat文件
data = {
    'pt': pt,
    'wt': wt,
    'SSEc': SSEc,
    'NEcpt': NEcpt,
    'WEcpt': WEcpt
}

savemat('parameter.mat', data)

# # 读取.mat文件
# # loaded_data = loadmat('parameter.mat')

# # 提取各个矩阵和数组
# # pt_loaded = loaded_data['pt']
# # t_loaded = loaded_data['wt']
# # Sec_loaded = loaded_data['Sec']
# # NEcpt_loaded = loaded_data['NEcpt']
# # Ecpt_loaded = loaded_data['Ecpt']
#
# # 打印以确认读取正确
print("Preparation Time Matrix (pt):\n", pt)
print("Processing Time Matrix (t):\n", wt)
print("Start/Stop Energy Consumption (Sec):\n", SSEc)
print("No-load Energy Consumption Coefficient (NEcpt):\n", NEcpt)
print("Energy Consumption Coefficient (WEcpt):\n", WEcpt)
