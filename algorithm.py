import numpy as np
from utils import bradley_terry_prob, get_fisher_info_duel
from math import log, sqrt

# 算法 1: EstimateBeta (参数 β 估计)

def estimate_beta(K, T_c, S_c, N, W, beta_init, sigma_sq, epsilon, alpha, max_iter):
    """
    算法 1: 使用自然梯度上升计算 β 的最大似然估计 (MLE)。
    ... (详细的参数和返回说明请参考原代码)
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
                g_i_duel += (W[i, j] - N[i, j] * p_ij) 
                
                # Fisher 信息 I_ii 的对决贡献 (Eq. 10)
                I_ii_duel += N[i, j] * get_fisher_info_duel(beta_hat[i], beta_hat[j])
            
            # 整合梯度和 Fisher 信息
            g[i] = g_i_classic + g_i_duel
            I_ii[i] = I_ii_classic + I_ii_duel

        # 3. 自然梯度更新 (Natural Gradient Update)
        # 避免除以零
        update = np.divide(g, I_ii, out=np.zeros_like(g, dtype=float), where=I_ii != 0)
        beta_hat_new = beta_hat_old + alpha * update
        
        # 4. 收敛检查
        if np.max(np.abs(beta_hat_new - beta_hat_old)) < epsilon:
            beta_hat = beta_hat_new
            break
            
        beta_hat = beta_hat_new
    
    return beta_hat

# 算法 2: Hybrid BAI with Simultaneous Observations (混合同时观测的 BAI)

def hybrid_bai(K, delta, sigma_sq, r_mean, duel_matrix, beta_init, epsilon, alpha, max_iter, sim_rounds=1000):
    """
    算法 2: 混合同时观测的最佳臂识别 (Hybrid BAI with Simultaneous Observations)。
    ... (详细的参数和返回说明请参考原代码)
    """
    
    # 1. 初始化统计量
    T_c = np.zeros(K, dtype=int)      # 经典观测计数 T_i^(c)
    S_c = np.zeros(K, dtype=float)    # 经典观测奖励总和 S_i^(c)
    N = np.zeros((K, K), dtype=int)   # 对决总次数 N_ij
    W = np.zeros((K, K), dtype=int)   # 胜利次数 W_ij

    # 预计算 Bonferroni 校正因子 z
    # z = sqrt(2 * log(2*K / delta))
    z_factor = sqrt(2.0 * log(2.0 * K / delta))
    
    print(f"开始 Hybrid BAI 模拟 (K={K}, delta={delta})")

    for t in range(1, sim_rounds + 1):
        # 2. 参数估计
        beta_hat = estimate_beta(K, T_c, S_c, N, W, beta_init, sigma_sq, epsilon, alpha, max_iter)
        if t % 1000 == 0:
            print(f"回合 {t}: 当前 β 估计: {beta_hat}")
        # 为了计算 I_ii，需要重新运行 EstimateBeta 的部分逻辑 (Fisher 信息计算)
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
        
        # 仅对已观测的臂计算置信区间
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
        
        # A. 经典观测臂选择 (i_c): 使用 D-最优准则的近似 (最大化 Fisher 信息增益)
        IG_classic = np.zeros(K)
        for i in range(K):
            # 假设 T_c[i] + 1
            I_ii_classic_new = (T_c[i] + 1) / sigma_sq
            # 经典项更新，对决项不变
            I_ii_new = I_ii_classic_new + (I_ii[i] - T_c[i] / sigma_sq) 
            
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
                # 增益近似: 最大化 $\log(I_{jj}^{new}) - \log(I_{jj}) + \log(I_{kk}^{new}) - \log(I_{kk})$
                F_jk = get_fisher_info_duel(beta_hat[j], beta_hat[k])
                
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