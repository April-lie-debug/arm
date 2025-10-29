from utils import *

# 设定参数
delta = 0.05       # 失败概率
sigma_sq = 1.0     # 奖励方差 σ²

# EstimateBeta 参数
epsilon = 1e-4     # 收敛容忍度
alpha = 1          # 学习率
max_iter = 100     # EstimateBeta 的最大迭代次数

# 模拟环境的真实参数 (用于生成数据)
# β 参数 (对数强度): 决定奖励均值和对决概率
true_beta = np.array([0.05, 0.1, 0.15, 0.2, 0.25, .3, .35, .4, .45, .5])
K = len(true_beta)
true_r_mean = true_beta # 经典观测: r_i ~ N(β_i, σ²)
true_duel_matrix = true_beta # 对决观测: Bradley-Terry 模型基于 β
