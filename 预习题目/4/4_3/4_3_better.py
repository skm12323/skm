# import os
# 可选：指定使用的 GPU（若有多个）
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import tensorcircuit as tc
import matplotlib.pyplot as plt

# ===================== 1. 设置后端和随机种子 =====================
tc.set_backend("jax")          # 使用 JAX 后端 (需已安装 jax[cuda] 或 jax[cpu])
#tc.set_random_seed(42)         # 固定随机种子，保证可复现

# ===================== 2. 构建 Bell 态电路 =====================
c = tc.Circuit(2)
c.h(0)
c.cx(0, 1)
exact = c.expectation_ps(z=[0, 1])  # 精确值应为 1.0
print(f"精确期望值: {exact:.6f}")

# ===================== 3. 定义采样估计函数 =====================
def compute(N):
    """
    通过 N 次采样估计 <Z0 Z1>
    """
    # sample 返回形状为 (N,) 的整数数组，0~3 对应 |00>,|01>,|10>,|11>
    samples = c.sample(batch=N, format="sample_int")
    # 将样本映射为 ±1：|00> 或 |11> 对应 +1，其他对应 -1
    signs = np.where((samples == 0) | (samples == 3), 1, -1)
    return np.mean(signs)  # 直接返回均值

def diff(N):
    """估计误差"""
    return compute(N) - exact

# ===================== 4. 生成样本数序列 =====================
Ns = np.logspace(1, 5, 30, dtype=int)  # 从 10 到 100,000，共 30 个点

# ===================== 5. 重复实验，计算误差统计 =====================
repeats = 50                    # 每个 N 重复采样的次数
diffs_mean = []
diffs_std = []

for N in Ns:
    errs = [diff(N) for _ in range(repeats)]
    diffs_mean.append(np.mean(errs))
    diffs_std.append(np.std(errs))

# ===================== 6. 绘图 =====================
plt.figure(figsize=(8, 5))
plt.semilogx(Ns, diffs_mean, 'o-', color='b', label='Mean error')
plt.fill_between(
    Ns,
    np.array(diffs_mean) - np.array(diffs_std),
    np.array(diffs_mean) + np.array(diffs_std),
    alpha=0.2, color='b', label='±1 std'
)

# 理论参考线：~1/√N (缩放系数仅为视觉对比)
theory = 0.5 / np.sqrt(Ns)
plt.semilogx(Ns, theory, '--', color='r', label=r'~$1/\sqrt{N}$ (scaled)')

plt.xlabel('Sample size $N$ (log scale)')
plt.ylabel('Estimation error')
plt.title('Sampling error of $\langle Z_0 Z_1 \\rangle$ for Bell state')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# 可选：打印后端信息确认 GPU 是否被使用
# tc.about()