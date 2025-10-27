import numpy as np
from math import log, exp, sqrt
from scipy.optimize import minimize_scalar

# 设置随机种子以保证结果可复现性（可选）
np.random.seed(42)

# =================================================================
# 辅助函数
# =================================================================

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

# =================================================================
# 算法 1: EstimateBeta (参数 β 估计)
# =================================================================

def estimate_beta(K, T_c, S_c, N, W, beta_init, sigma_sq, epsilon, alpha, max_iter):
    """
    算法 1: 使用自然梯度上升计算 β 的最大似然估计 (MLE)。

    参数:
    K (int): 臂的数量。
    T_c (np.ndarray): 经典观测计数 [K]。
    S_c (np.ndarray): 经典观测奖励总和 [K]。
    N (np.ndarray): 对决总次数 N_ij [K x K]。
    W (np.ndarray): 臂 i 击败臂 j 的次数 W_ij [K x K]。
    beta_init (np.ndarray): 初始 β 估计 [K]。
    sigma_sq (float): 经典观测的奖励方差 σ²。
    epsilon (float): 收敛容忍度 ε。
    alpha (float): 学习率 α。
    max_iter (int): 最大迭代次数。

    返回:
    np.ndarray: MLE 估计 β_hat [K]。
    """
    beta_hat = np.array(beta_init, dtype=float)
    
    for iter_count in range(max_iter):
        beta_hat_old = beta_hat.copy()
        g = np.zeros(K)
        I_ii = np.zeros(K)
        
        # 1. 计算均值奖励 (Compute Mean Rewards)
        r_bar = np.divide(S_c, T_c, out=np.zeros_like(S_c, dtype=float), where=T_c != 0)
        
        for i in range(K):
            # 初始化经典观测贡献
            I_ii_classic = T_c[i] / sigma_sq
            g_i_classic = I_ii_classic * (r_bar[i] - beta_hat[i])

            # 2. 计算梯度 g_i 和 Fisher 信息 I_ii 的对决贡献
            g_i_duel = 0.0
            I_ii_duel = 0.0
            
            for j in range(K):
                if i == j:
                    continue
                
                # 计算 P(i > j | beta_hat)
                p_ij = bradley_terry_prob(beta_hat[i], beta_hat[j])
                
                # 梯度 g_i 的对决贡献 (Eq. 5)
                # 注意：论文中的公式 W_ij - N_ij * p_ij 对应的是臂 i 击败 j 的观测和期望
                g_i_duel += (W[i, j] - N[i, j] * p_ij) 
                
                # Fisher 信息 I_ii 的对决贡献 (Eq. 10)
                I_ii_duel += N[i, j] * get_fisher_info_duel(beta_hat[i], beta_hat[j])
            
            # 整合梯度和 Fisher 信息
            g[i] = g_i_classic + g_i_duel
            I_ii[i] = I_ii_classic + I_ii_duel

        # 3. 自然梯度更新 (Natural Gradient Update)
        # 避免除以零：如果 I_ii_i = 0，则该臂没有被观测，梯度也应为 0
        update = np.divide(g, I_ii, out=np.zeros_like(g, dtype=float), where=I_ii != 0)
        beta_hat_new = beta_hat_old + alpha * update
        
        # 4. 收敛检查
        if np.max(np.abs(beta_hat_new - beta_hat_old)) < epsilon:
            beta_hat = beta_hat_new
            break
            
        beta_hat = beta_hat_new
    
    return beta_hat

# =================================================================
# 算法 2: Hybrid BAI with Simultaneous Observations (混合同时观测的 BAI)
# =================================================================

def hybrid_bai(K, delta, sigma_sq, r_mean, duel_matrix, beta_init, epsilon, alpha, max_iter, sim_rounds=1000):
    """
    算法 2: 混合同时观测的最佳臂识别 (Hybrid BAI with Simultaneous Observations)。
    
    参数:
    K (int): 臂的数量。
    delta (float): 失败概率 δ。
    sigma_sq (float): 经典观测的奖励方差 σ²。
    r_mean (np.ndarray): 真实奖励均值 (用于模拟观测) [K]。
    duel_matrix (np.ndarray): 真实 Bradley-Terry 模型参数 (用于模拟观测) [K]。
    beta_init, epsilon, alpha, max_iter: 传递给 EstimateBeta 的参数。
    sim_rounds (int): 模拟的最大回合数。

    返回:
    int: 识别出的最佳臂索引 (从 0 开始)。
    """
    
    # 1. 初始化统计量
    T_c = np.zeros(K, dtype=int)        # 经典观测计数 T_i^(c)
    S_c = np.zeros(K, dtype=float)      # 经典观测奖励总和 S_i^(c)
    N = np.zeros((K, K), dtype=int)     # 对决总次数 N_ij
    W = np.zeros((K, K), dtype=int)     # 胜利次数 W_ij

    # 预计算 Bonferroni 校正因子 z
    # z = sqrt(2 * log(2*K / delta))
    z_factor = sqrt(2.0 * log(2.0 * K / delta))
    
    print(f"开始 Hybrid BAI 模拟 (K={K}, delta={delta})")

    for t in range(1, sim_rounds + 1):
        # 2. 参数估计
        beta_hat = estimate_beta(K, T_c, S_c, N, W, beta_init, sigma_sq, epsilon, alpha, max_iter)
        
        # 为了计算 I_ii，需要重新运行 EstimateBeta 的部分逻辑
        I_ii = np.zeros(K)
        for i in range(K):
            I_ii_classic = T_c[i] / sigma_sq
            I_ii_duel = 0.0
            for j in range(K):
                if i == j: continue
                p_ij = bradley_terry_prob(beta_hat[i], beta_hat[j])
                I_ii_duel += N[i, j] * get_fisher_info_duel(beta_hat[i], beta_hat[j])
            I_ii[i] = I_ii_classic + I_ii_duel

        # 3. 计算置信区间和停止检查
        L = np.zeros(K)
        U = np.zeros(K)
        
        # 仅对已观测的臂计算置信区间，未观测的臂 I_ii=0，导致 L/U 默认无穷大
        for i in range(K):
            if I_ii[i] > 0:
                CI_width = z_factor / sqrt(I_ii[i])
                L[i] = beta_hat[i] - CI_width
                U[i] = beta_hat[i] + CI_width
            else:
                L[i] = -np.inf
                U[i] = np.inf
        
        i_star = np.argmax(beta_hat) # 经验最佳臂

        # 停止条件检查: L_i* > max_{j!=i*} U_j
        max_U_not_i_star = -np.inf
        if K > 1:
            indices_not_i_star = [j for j in range(K) if j != i_star]
            max_U_not_i_star = np.max(U[indices_not_i_star])
            
        if L[i_star] > max_U_not_i_star:
            print(f"在回合 {t} 停止。识别出的最佳臂索引: {i_star}")
            return i_star

        # 4. 信息增益计算和观测选择
        
        # A. 经典观测臂选择 (i_c): 最大化 IG_classic(i) = T_i^(c)^2 * (∂²l / ∂T_i^(c)²)^-1
        # 在对角近似下，IG_classic(i) ∝ 1/I_ii_classic = σ²/T_i^(c)
        # 或者更简单地，使用 Eq. (15) 的形式，与 I_ii_classic 最小化相关
        # 论文 Eq. (15): IG_classic(i) = (1/2) * (1 / (1/σ² + 1/(σ²*T_i^(c)))) - T_i^(c) * (∂²l_c / ∂T_i^(c)²)^-1
        # 然而，在独立选择步骤中，通常使用 D-最优准则 (最大化 I_ii) 或近似
        # 此处我们遵循论文精神，即选择能最大化 Fisher 信息的臂
        
        # 根据 D-最优准则 (最大化 det(I)) 的对角近似，选择使 I_ii 增益最大的
        IG_classic = np.zeros(K)
        for i in range(K):
            # 新的 Fisher 信息 I_ii_new (假设 T_c[i] + 1)
            I_ii_classic_new = (T_c[i] + 1) / sigma_sq
            I_ii_new = I_ii_classic_new + (I_ii[i] - T_c[i] / sigma_sq) # 经典项更新，对决项不变
            
            # 信息增益近似: log(I_ii_new) - log(I_ii)
            if I_ii[i] > 0:
                 IG_classic[i] = np.log(I_ii_new) - np.log(I_ii[i])
            else:
                 # 未观测的臂，增益无穷大 (应优先选择)
                 IG_classic[i] = np.inf
        
        i_c = np.argmax(IG_classic) # 经典观测臂
        
        # B. 对决观测对选择 (j_d, k_d)
        IG_duel = -np.inf
        j_d, k_d = -1, -1
        
        for j in range(K):
            for k in range(j + 1, K):
                # 对决观测信息增益近似: 
                # 选择使 det(I) 增益最大的对。在对角近似下，选择使 I_jj + I_kk 增益最大的
                # 增益 ∝ 1/2 * log( (I_jj + N_{jk}*F_{jk}) / I_jj ) + 1/2 * log( (I_kk + N_{jk}*F_{jk}) / I_kk )
                # 论文 Eq. (20) 简化形式: IG_duel(j, k) = (1/2) * (1/I_jj + 1/I_kk) * N_{jk} * F_{jk}
                
                # 为简化，我们选择能最大化 I_jj + I_kk 增益的对 (这是 D-最优准则的合理近似)
                F_jk = get_fisher_info_duel(beta_hat[j], beta_hat[k])
                
                # 假设进行一次 (j, k) 对决，N[j, k] 变为 N[j, k]+1, N[k, j] 变为 N[k, j]+1
                # I_ii 增益: N_{jk}*F_{jk} 增加 F_{jk}
                
                current_ig = 0.0
                if I_ii[j] > 0 and I_ii[k] > 0:
                    I_jj_new = I_ii[j] + F_jk
                    I_kk_new = I_ii[k] + F_jk
                    current_ig = (np.log(I_jj_new) - np.log(I_ii[j])) + \
                                 (np.log(I_kk_new) - np.log(I_ii[k]))
                else:
                    # 如果任何一个臂的 I_ii=0，优先选择 (应先观测)
                    current_ig = np.inf 
                    
                if current_ig > IG_duel:
                    IG_duel = current_ig
                    j_d, k_d = j, k
        
        # 5. 执行同时观测
        
        # A. 经典观测: 臂 i_c
        # 模拟奖励 r ~ N(r_mean[i_c], sigma_sq)
        r_i_c = np.random.normal(r_mean[i_c], sqrt(sigma_sq))
        T_c[i_c] += 1
        S_c[i_c] += r_i_c
        
        # B. 对决观测: 臂对 (j_d, k_d)
        # 模拟对决结果 w (j_d 击败 k_d)
        p_jd_kd = bradley_terry_prob(duel_matrix[j_d], duel_matrix[k_d])
        w = np.random.rand() < p_jd_kd # True if j_d wins
        
        N[j_d, k_d] += 1
        N[k_d, j_d] += 1
        
        if w: # j_d wins k_d
            W[j_d, k_d] += 1
        else: # k_d wins j_d
            W[k_d, j_d] += 1
            
    print(f"达到最大回合数 {sim_rounds}。返回当前经验最佳臂 {i_star}")
    return i_star

# =================================================================
# 示例用法 (Simulation Example)
# =================================================================
if __name__ == '__main__':
    # 设定参数
    K = 5               # 臂的数量
    delta = 0.05        # 失败概率
    sigma_sq = 1.0      # 奖励方差 σ²
    
    # EstimateBeta 参数
    epsilon = 1e-4      # 收敛容忍度
    alpha = 1         # 学习率
    max_iter = 100      # EstimateBeta 的最大迭代次数
    
    # 模拟环境的真实参数 (用于生成数据)
    # β 参数 (对数强度): 决定奖励均值和对决概率
    true_beta = np.array([0.1,0.2,0.3,0.4,0.5]) 
    true_r_mean = true_beta # 经典观测: r_i ~ N(β_i, σ²)
    true_duel_matrix = true_beta # 对决观测: Bradley-Terry 模型基于 β
    
    # 初始 β 估计
    initial_beta_hat = np.zeros(K) 

    # 运行 Hybrid BAI 算法
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
        sim_rounds=1e5 
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