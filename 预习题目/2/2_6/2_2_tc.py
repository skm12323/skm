# 2-2 矩阵指数 —— 使用 TensorCircuit 后端 (jax) 重写
import numpy as np
import tensorcircuit as tc

tc.set_dtype("complex128")
K = tc.set_backend("jax")

sigma_x = K.stack([K.stack([0, 1]), K.stack([1, 0])])
sigma_y = K.stack([K.stack([0, -1j]), K.stack([1j, 0])])
sigma_z = K.stack([K.stack([1, 0]), K.stack([0, -1])])
I_2 = K.stack([K.stack([1, 0]), K.stack([0, 1])])

#sigma_x = K.convert_to_tensor(np.array([[0, 1], [1, 0]], dtype=complex))
#sigma_y = K.convert_to_tensor(np.array([[0, -1j], [1j, 0]], dtype=complex))
#sigma_z = K.convert_to_tensor(np.array([[1, 0], [0, -1]], dtype=complex))
#I_2     = K.convert_to_tensor(np.array([[1, 0], [0, 1]], dtype=complex))

def pauli_exp_diff(P, theta):
    exp_val = K.expm(1j * theta * P)
    compute_exp_val = K.cos(theta) * I_2 + 1j * K.sin(theta) * P
    diff_exp = exp_val - compute_exp_val
    return diff_exp

if __name__ == "__main__":
    pauli = {"x": sigma_x, "y": sigma_y, "z": sigma_z}

    axis = input("请输入泡利矩阵 (x/y/z): ").strip().lower()
    if axis not in pauli:
        raise ValueError(f"无效的轴 '{axis}'，只能输入 x、y 或 z")

    coeff = float(input("请输入 theta 为多少倍 pi: "))
    theta = coeff * np.pi

    print(f"\nsigma_{axis} (theta = {coeff} * pi):")
    print(pauli_exp_diff(pauli[axis], theta))