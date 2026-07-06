import numpy as np
from scipy.linalg import expm

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)


def pauli_exp_diff(P, theta):
    exp_val = expm(1j * theta * P)
    compute_exp_val = np.cos(theta) * np.eye(2, dtype=complex) + 1j * np.sin(theta) * P
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
