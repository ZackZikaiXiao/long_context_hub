import torch
import torch.nn.functional as F
from sklearn.metrics import mutual_info_score

def load_probs(file_path):
    return torch.load(file_path)

def calculate_kl_divergence(probs_with, probs_without):
    # 在对概率取对数之前，添加一个小的正常数以避免对零取对数
    eps = 1e-9
    probs_with_safe = torch.clamp(probs_with, min=eps)  # 确保概率值不小于eps
    probs_without_safe = torch.clamp(probs_without, min=eps)
    kl_div = F.kl_div(probs_with_safe.log(), probs_without_safe, reduction='batchmean')
    return kl_div.item()

def calculate_mutual_information(probs_with, probs_without):
    # 转换为numpy数组进行互信息计算
    probs_with_np = probs_with.squeeze().numpy()  # 使用squeeze去除多余的维度
    probs_without_np = probs_without.squeeze().numpy()
    mi = mutual_info_score(probs_with_np, probs_without_np)
    return mi

def calculate_cosine_similarity(probs_with, probs_without):
    cos_sim = F.cosine_similarity(probs_with, probs_without, dim=0)
    return cos_sim.item()

def calculate_absolute_difference(probs_with, probs_without):
    abs_diff = torch.sum(torch.abs(probs_with - probs_without))
    return abs_diff.item()

# 加载prob文件
probs_0 = load_probs('./next_token_base.pt').squeeze().cpu().detach()  # 使用squeeze
probs_1 = load_probs('./next_token_base.pt').squeeze().cpu().detach()
probs_2 = load_probs('./next_token_base.pt').squeeze().cpu().detach()
probs_3 = load_probs('./next_token_base.pt').squeeze().cpu().detach()

probs_0_init = load_probs('./next_token_kv_cache_0_init.pt').squeeze().cpu().detach()
probs_1_init = load_probs('./next_token_kv_cache_1_init.pt').squeeze().cpu().detach()
probs_2_init = load_probs('./next_token_kv_cache_2_init.pt').squeeze().cpu().detach()
probs_3_init = load_probs('./next_token_kv_cache_3_init.pt').squeeze().cpu().detach()

# 计算变化量
# Prompt 0
kl_div_0 = calculate_kl_divergence(probs_0_init, probs_0)
mi_0 = calculate_mutual_information(probs_0_init, probs_0)
cos_sim_0 = calculate_cosine_similarity(probs_0_init, probs_0)
abs_diff_0 = calculate_absolute_difference(probs_0_init, probs_0)

# Prompt 1
kl_div_1 = calculate_kl_divergence(probs_1_init, probs_1)
mi_1 = calculate_mutual_information(probs_1_init, probs_1)
cos_sim_1 = calculate_cosine_similarity(probs_1_init, probs_1)
abs_diff_1 = calculate_absolute_difference(probs_1_init, probs_1)

# Prompt 2
kl_div_2 = calculate_kl_divergence(probs_2_init, probs_2)
mi_2 = calculate_mutual_information(probs_2_init, probs_2)
cos_sim_2 = calculate_cosine_similarity(probs_2_init, probs_2)
abs_diff_2 = calculate_absolute_difference(probs_2_init, probs_2)

# Prompt 3
kl_div_3 = calculate_kl_divergence(probs_3_init, probs_3)
mi_3 = calculate_mutual_information(probs_3_init, probs_3)
cos_sim_3 = calculate_cosine_similarity(probs_3_init, probs_3)
abs_diff_3 = calculate_absolute_difference(probs_3_init, probs_3)

print(f'Prompt 0 - KL Divergence: {kl_div_0}, Mutual Information: {mi_0}, Cosine Similarity: {cos_sim_0}, Absolute Difference: {abs_diff_0}')
print(f'Prompt 1 - KL Divergence: {kl_div_1}, Mutual Information: {mi_1}, Cosine Similarity: {cos_sim_1}, Absolute Difference: {abs_diff_1}')
print(f'Prompt 2 - KL Divergence: {kl_div_2}, Mutual Information: {mi_2}, Cosine Similarity: {cos_sim_2}, Absolute Difference: {abs_diff_2}')
print(f'Prompt 3 - KL Divergence: {kl_div_3}, Mutual Information: {mi_3}, Cosine Similarity: {cos_sim_3}, Absolute Difference: {abs_diff_3}')
