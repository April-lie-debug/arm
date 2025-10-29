import numpy as np
from algorithm import estimate_beta, hybrid_bai # 导入算法函数

# 设置随机种子以保证结果可复现性（可选）
np.random.seed(42)

if __name__ == '__main__':
    # 设定参数
    K = 5              # 臂的数量
    delta = 0.05       # 失败概率
    sigma_sq = 1.0     # 奖励方差 σ²
    
    # EstimateBeta 参数
    epsilon = 1e-4     # 收敛容忍度
    alpha = 1          # 学习率
    max_iter = 100     # EstimateBeta 的最大迭代次数
    
    # 模拟环境的真实参数 (用于生成数据)
    # β 参数 (对数强度): 决定奖励均值和对决概率
    true_beta = np.array([0.1, 0.2, 0.3, 0.4, 0.5]) 
    true_r_mean = true_beta # 经典观测: r_i ~ N(β_i, σ²)
    true_duel_matrix = true_beta # 对决观测: Bradley-Terry 模型基于 β
    
    # 初始 β 估计
    initial_beta_hat = np.zeros(K) 

    # 运行 Hybrid BAI 算法
    sim_rounds = 1e5 
    identified_best_arm = hybrid_bai(
        K=K, 
        delta=delta, 
        sigma_sq=sigma_sq, 
        r_mean=true_r_mean, 
        duel_matrix=true_duel_matrix, 
        beta_init=initial_beta_hat, 
        epsilon=epsilon, 
        alpha=alpha, 
        max_iter=max_iter,
        sim_rounds=int(sim_rounds) # 确保 sim_rounds 是整数
    )

    # 真实最佳臂
    true_best_arm = np.argmax(true_beta)
    
    print("\n--- 模拟结果 ---")
    print(f"真实 β 参数: {true_beta}")
    print(f"真实最佳臂索引: {true_best_arm}")
    print(f"算法识别出的最佳臂索引: {identified_best_arm}")
    if identified_best_arm == true_best_arm:
        print("识别正确。")
    else:
        print("识别错误。")