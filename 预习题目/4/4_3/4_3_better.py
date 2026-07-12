import os
import numpy as np
import tensorcircuit as tc
import matplotlib.pyplot as plt

# 2 比特问题用 numpy 后端即可：无需 GPU，也避开 jax 在小算子上的派发开销。
tc.set_backend("numpy")

# 原先改为使用 |+0⟩=(|00⟩+|10⟩)/√2（不加 cx）：P(偶)=P(奇)=0.5，估计量标准差 = 1/√N。
# Bell 态（加 cx）下 P(奇)=0、方差为 0，看不到抽样噪声。
c = tc.Circuit(2)
c.h(0)
c.h(1)  # 题目修改为增加对1施加 Hadamard 门，为(|00>+|01>+|10>+|11>)/2。
# c.cx(0, 1)   # 加上是 Bell 态

probs = np.abs(np.array(c.state()).flatten()) ** 2          # 末态 4 个概率 |amp|^2
exact = float(np.real(c.expectation_ps(z=[0, 1])))          # 精确 <Z0Z1>
print(f"probs = {np.round(probs, 4)},  精确 <Z0Z1> = {exact}")

rng = np.random.default_rng(0)

def compute(N):
    # 从精确分布抽 N 个 iid 样本，统计上等价于 c.sample，但纯用CPU无派发开销。
    s = rng.choice(4, size=N, p=probs)
    return float(np.mean(np.where((s == 0) | (s == 3), 1, -1)))
def diff(N):
    return compute(N) - exact

Ns = np.logspace(1, 5, 30, dtype=int)
repeats = 20

# 用 RMS 误差
rms_err = [
    np.sqrt(np.mean(np.square([diff(N) for _ in range(repeats)]))) for N in Ns
]

plt.figure(figsize=(8, 5))
plt.semilogx(Ns, rms_err, "o-", color="b", label=f"RMS error ({repeats} repeats)")
plt.semilogx(Ns, 1.0 / np.sqrt(Ns), "--", color="r", label=r"$1/\sqrt{N}$")

plt.xlabel("Sample size $N$ (log scale)")
plt.ylabel(r"Estimation error of $\langle Z_0 Z_1\rangle$")
plt.title(r"Sampling error vs $N$ for $|+0\rangle$ state")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

# 存到脚本同目录，且不覆盖 4_3.py 的 expectations_diff.png
out_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "expectations_diff_h_01.png",
)# 原仅对0施加H门的图为 expectations_diff_better.png
plt.savefig(out_path, dpi=200)
print(f"Saved plot to {out_path}")
plt.show()

# ── 通用写法（大线路才需要）──
# 几十上百比特时拿不到完整概率分布，才用 TC 采样器
# 一次 batch 抽完，避免 600 次小调用带来的派发开销：
#   samples = c.sample(batch=repeats * N, format="sample_int").reshape(repeats, N)
#   errs = np.mean(np.where((samples == 0) | (samples == 3), 1, -1), axis=1) - exact
