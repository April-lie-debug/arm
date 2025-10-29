import numpy as np
from algorithm import estimate_beta, hybrid_bai # 导入算法函数
from params import *

# 设置随机种子以保证结果可复现性（可选）
np.random.seed(42)

if __name__ == '__main__':
    # 运行 Hybrid BAI 算法
    # 初始 β 估计
    initial_beta_hat = np.zeros(K) 
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