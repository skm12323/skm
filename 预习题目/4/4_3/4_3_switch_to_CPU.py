# 演示：用 CUDA_VISIBLE_DEVICES 把 jax 后端切到 CPU（TensorCircuit FAQ 的 GPU/CPU 切换法）。
#
# ★ 关键 ★：必须在 import tensorcircuit / jax 之前设置该环境变量，
#            否则 jax 在 import 时就已经抓取了 GPU，再设无效。
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # "" = 隐藏所有 GPU(强制 CPU)；"0" = 用 0 号 GPU；不设 = 默认(GPU 优先)

import time
import numpy as np
import tensorcircuit as tc
import matplotlib.pyplot as plt

tc.set_backend("jax")      # 仍是 jax 后端，但因上面隐藏了 GPU，实际跑在 CPU 上

# ---- 验证确实切到了 CPU ----
import jax
print("jax devices:", jax.devices())   # 期望：仅 CpuDevice，没有 GPU

# ---- 同一个 |+0⟩ 电路，仍用 TC 的 c.sample（对照之前 GPU 上 thrash 的版本）----
c = tc.Circuit(2)
c.h(0)
exact = float(np.real(c.expectation_ps(z=[0, 1])))
print(f"精确 <Z0Z1> = {exact}")

def compute(N):
    s = np.array(c.sample(batch=N, format="sample_int"))
    return float(np.mean(np.where((s == 0) | (s == 3), 1, -1)))

# ---- 单次计时参考（c.sample 在 CPU 上的单次开销）----
t = time.time(); compute(10000); print(f"1x c.sample(10000): {time.time() - t:.3f}s")

# ---- RMS 扫描（c.sample 较慢，N 只到 1e4、重复 5 次以控制总耗时）----
Ns = np.logspace(1, 4, 12, dtype=int)
repeats = 5

t = time.time()
rms_err = [np.sqrt(np.mean(np.square([compute(N) - exact for _ in range(repeats)]))) for N in Ns]
sweep_time = time.time() - t
print(f"CPU c.sample 全扫描({len(Ns)} N × {repeats} 重复)耗时: {sweep_time:.2f}s")

# ---- 画图 ----
plt.figure(figsize=(8, 5))
plt.semilogx(Ns, rms_err, "o-", color="g", label=f"RMS error ({repeats} repeats, c.sample on CPU)")
plt.semilogx(Ns, 1.0 / np.sqrt(Ns), "--", color="r", label=r"$1/\sqrt{N}$")
plt.xlabel("Sample size $N$ (log scale)")
plt.ylabel(r"Estimation error of $\langle Z_0 Z_1\rangle$")
plt.title(r"Sampling error vs $N$ (jax backend forced to CPU)")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()

out_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "expectations_diff_cpu.png",   # 不覆盖 expectations_diff.png / expectations_diff_better.png
)
plt.savefig(out_path, dpi=200)
print(f"Saved plot to {out_path}")
plt.show()
