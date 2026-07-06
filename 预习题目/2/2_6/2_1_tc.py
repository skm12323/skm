# 2-1 旋转矩阵 —— 使用 TensorCircuit 后端 (jax) 重写
import numpy as np
import tensorcircuit as tc

K = tc.set_backend("jax")

theta = K.convert_to_tensor(np.pi / 4)  # 旋转角度（顺时针）

R = K.stack([
    K.stack([K.cos(theta),  K.sin(theta)]),
    K.stack([-K.sin(theta), K.cos(theta)]),
])

v = K.convert_to_tensor([1., 0.])

v_1 = K.tensordot(R, v, axes=1)

print(K.numpy(v_1))


# ──────────────────────────────────────────────────────────────
# v=(1,0) 被 R(theta) 顺时针旋转，端点扫出单位圆。
import matplotlib.pyplot as plt

def rotate(t):
    Rr = K.stack([
        K.stack([K.cos(t),  K.sin(t)]),
        K.stack([-K.sin(t), K.cos(t)]),
    ])
    return K.tensordot(Rr, K.convert_to_tensor([1., 0.]), axes=1)

thetas = K.convert_to_tensor(np.linspace(0, 2 * np.pi, 200))
vs = K.numpy(K.vmap(rotate)(thetas))   # shape (200, 2)

plt.figure(figsize=(4, 4))
plt.plot(vs[:, 0], vs[:, 1])
plt.scatter([1.0], [0.0], color="red")      # 起点 theta=0
plt.gca().set_aspect("equal")
plt.xlabel("x"); plt.ylabel("y")
plt.title(r"$R(\theta)\,(1,0)$")
plt.grid(True)
plt.show()
