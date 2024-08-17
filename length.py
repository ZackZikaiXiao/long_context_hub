import random
import json
import numpy as np
import matplotlib.pyplot as plt

data_path = "./dataset/alpaca_data_pre.json"
data_path = "./data_augument/eda_LongAlpaca-12k.json"

with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

num_samples = int(len(data) * 1)
sampled_indices = random.sample(range(len(data)), num_samples)
sampled_data = [data[i] for i in sampled_indices]
data = sampled_data

sample_lengths = np.array([len(sample['instruction'].replace(' ', '') + sample['output'].replace(' ', '')) for sample in data])

# 将样本长度进行分段（分箱）
num_bins = 100
min_length = sample_lengths.min()
max_length = sample_lengths.max()
bins = np.linspace(min_length, max_length, num_bins + 1)
binned_lengths = np.digitize(sample_lengths, bins) - 1  # 得到每个样本所属的bin索引
# 统计每个bin的频率
bin_freq = np.bincount(binned_lengths, minlength=num_bins)

# 创建图形
plt.figure(figsize=(10, 6))
# 绘制柱状图
plt.bar(range(len(bin_freq)), bin_freq)
# 添加标题和标签
plt.title('Frequency Distribution')
plt.xlabel('Bins')
plt.ylabel('Frequency')
# 显示网格
plt.grid(True)
# 保存图像到本地
plt.savefig('eda_bin_freq_distribution.png')



# 防止 bin_freq 中包含0
bin_freq = np.where(bin_freq == 0, 1e-6, bin_freq)
# 计算每个bin频率的平方根
sqrt_freq = np.power(bin_freq, 0.25) / bin_freq
# 归一化平方根频率
total_sqrt_freq = np.sum(sqrt_freq)
normalized_probabilities = sqrt_freq / total_sqrt_freq
# 分配每个样本的权重
sample_weights = normalized_probabilities[binned_lengths]