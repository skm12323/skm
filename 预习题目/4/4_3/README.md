# 4-3 说明

本目录三个对照脚本，都围绕**用抽样估计 ⟨Z₀Z₁⟩ 的误差随样本数 N 的变化**，但分别演示不同现象/方法。

## 三个文件

| 文件 | 电路 | 采样方式 | 产物图 | 展示的现象 |
|---|---|---|---|---|
| `4_3.py` | Bell 态（H+CX） | `c.sample()` + 循环 `+= 1/N` | `expectations_diff.png` | 误差**随 N 增大**（反常）|
| `4_3_better.py` | \|+0⟩（只 H） | numpy 多项采样 | `expectations_diff_better.png` | 误差按 **1/√N** 下降（正常）|
| `4_3_switch_to_CPU.py` | \|+0⟩（只 H） | `c.sample()`，jax 强制 CPU | `expectations_diff_cpu.png` | 同上，演示 FAQ 的 GPU/CPU 切换 |

## `4_3.py` 的"反常"现象及根因

Bell 态下 `diff(N)` 在 N≳1e3 后**反而随 N 增长**，y 轴 ~1e-12。这不是物理，是两点叠加：

1. **Bell 态没有抽样噪声**：P(奇宇称)=0，符号恒 +1，估计量方差为 0，理论 diff 应恒为 0。
2. **`+= 1/N` 的浮点累加漂移**：循环累加 N 次的舍入误差随 N 增大；平时被统计噪声淹没，Bell 态把噪声清零后才暴露。

→ 换 `np.mean` 或换个有真实噪声的态，漂移即消失。

## 时间比较

三脚本规模不完全相同，但量级差清晰：

| 脚本 | 后端 / 设备 | 规模 | 耗时 |
|---|---|---|---|
| `4_3.py` | jax / **GPU** | 30 N × 20，N≤1e5 | **>280s 跑不完**（GPU thrash）|
| `4_3_switch_to_CPU.py` | jax / **CPU** | 12 N × 5，N≤1e4 | **~13.5s**（单次 `c.sample(1e4)`≈1.06s）|
| `4_3_better.py` | **numpy** / CPU | 30 N × 20，N≤1e5 | **~0.16s** |

## 总结 / 选型

- **小线路不用 GPU**：2 比特任务有用计算量≈0，GPU 全忙在 kernel 派发开销上（FAQ明确说明了：≤16 比特 GPU 可能反而更慢）。
- **`CUDA_VISIBLE_DEVICES=""`**（须在 `import tensorcircuit` 前设）能让 jax 后端跑在 CPU 上，避免 GPU thrash 但 `c.sample` 本身仍慢（~1s/1e4 shot）。
- **纯采样小线路**：直接 numpy 后端 + `rng.choice` 多项采样，绕开 `c.sample`，最快。（不过我写 `4_3_better.py` 时还不知道可以换成仅 CPU ，不然我大概不会考虑尝试 `rng.choice` ）
- **需要 autodiff 的小线路**（如 §3）：用 jax + `CUDA_VISIBLE_DEVICES=""`，既避 thrash 又保留 `K.grad`。
- **大线路**（>16 比特）：需要 GPU + `c.sample`，且须 `batch=` 合并调用。
