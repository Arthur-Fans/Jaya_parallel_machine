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