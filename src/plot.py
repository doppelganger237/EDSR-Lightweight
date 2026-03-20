import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from adjustText import adjust_text

# 模型名
models = [
    "CARN", "IMDN", "RFDN", "MAFFSRN", "ECBSR", "RLFN", "ShuffleMixer", "PFDN"
]

# 对应 FLOPs (G)
flops = [90.9, 40.9, 31.6, 19.3, 34.73, 29.84, 28.0, 25.4]

# 对应 Params (K)
params = [1592, 715, 550, 441, 603, 543, 411, 468]

# 对应 PSNR (dB) —— Set14 ×4
psnr = [28.60, 28.58, 28.61, 28.58, 28.34, 28.62, 28.66, 28.83]

plt.figure(figsize=(6,5))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#17becf', '#e377c2']

max_size, min_size = 1600, 60
#sizes = [((p / min(params)) ** 0.5) * min_size for p in params]
sizes = [(p / min(params)) * min_size for p in params]
texts = []
for i, name in enumerate(models):
    x, y = flops[i], psnr[i]
    size = sizes[i]
    if name == "PFDN":
        plt.scatter(x, y, s=size, edgecolors='black', linewidths=0.8, color='red')
        texts.append(plt.text(x + 0.6, y , name, fontsize=10, color='red'))
    else:
        plt.scatter(x, y, s=size, edgecolors='black', linewidths=0.8, color=colors[i])
        texts.append(plt.text(x + 0.3, y + 0.01, name, fontsize=9))
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='none'))


# 坐标轴与标题
plt.xlabel("FLOPs (G)", fontsize=11)
plt.ylabel("PSNR (dB)", fontsize=11)
#plt.title("Performance–Complexity Trade-off", fontsize=12, pad=10)

# 样式
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# 保存与显示
plt.savefig("flops_psnr_set14_x4.png", dpi=600)
plt.show()