import numpy as np
from math import log, exp, sqrt

def bradley_terry_prob(beta_i, beta_j):
    """
    计算在 Bradley-Terry 模型下，臂 i 击败臂 j 的概率 P(i > j | beta)。
    P(i > j) = exp(beta_i) / (exp(beta_i) + exp(beta_j))
    """
    if beta_i == beta_j:
        return 0.5
    try:
        # 使用 log-sum-exp 技巧提高数值稳定性
        log_prob = beta_i - np.logaddexp(beta_i, beta_j)
        return np.exp(log_prob)
    except FloatingPointError:
        # 处理可能的溢出或下溢
        if beta_i > beta_j:
            return 1.0
        else:
            return 0.0

def get_fisher_info_duel(beta_i, beta_j):
    """
    计算单次对决观测 N_ij=1 带来的 Fisher 信息对角项贡献: p_ij * (1 - p_ij)
    """
    p_ij = bradley_terry_prob(beta_i, beta_j)
    return p_ij * (1.0 - p_ij)
