def compute(beta, p, w, q, l, sigma, k, gamma):
    return (beta / (k * gamma)) * (p * ((1 - q) + k * q) + (1 - p) * w * ((1 - q * l) + k * q * l))

def compute_beta(R0, p_v, w_v, q_v, l_v, sigma, k_v, gamma):
    return (R0 * k_v * gamma) / (p_v * ((1 - q_v) + k_v * q_v) + (1 - p_v) * w_v * ((1 - q_v * l_v) + k_v * q_v * l_v))

p_v = 0.9  # 0.1 0.5  0.9   # percent of unvaccinated
w_v = 0.5  # 感染率降低系数
beta = 0.6  # 感染率
q_v = 0.5  # 有症状的概率
l_v = 0.5  # 0-1 接种导致有症状概率降低系数
sigma_inverse = 5.2  # mean latent period
k_v = 0.5  # 0-1 无症状被发现的概率降低系数
gamma_inverse = 1.6  # mean 发现 quarantined period

R0 = compute(beta, p_v, w_v, q_v, l_v, 1 / sigma_inverse, k_v, 1 / gamma_inverse)
print("基本再生数R0：", R0)

R0 = 0.6
beta = compute_beta(R0, p_v, w_v, q_v, l_v, 1 / sigma_inverse, k_v, 1 / gamma_inverse)
print("基本感染率：", beta)

# R0 = (beta / (k_v * 1/gamma_inverse)) * (p_v * ((1 - q_v) + k_v * q_v) + (1 - p_v) * w_v * ((1 - q_v * l_v) + k_v * q_v * l_v)))
# beta = R0 * k_v / (p_v * ((1 - q_v) + k_v * q_v) + (1 - p_v) * w_v * ((1 - q_v * l_v) + k_v * q_v * l_v))

