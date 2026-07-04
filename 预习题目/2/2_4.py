import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def tensor_string(spec, n):
    """构造 n 个 2x2 矩阵的张量积"""
    ops = [I] * n
    for i, mat in spec.items():
        ops[i] = mat
    out = ops[0]
    for mat in ops[1:]:
        out = np.kron(out, mat)
    return out


def build_H(n):
    """ H = sum_{i=0}^{n-1} Z_i + sum_{i=0}^{n-2} X_i X_{i+1} """
    N = 2 ** n
    H = np.zeros((N, N), dtype=complex)
    for i in range(n):
        H += tensor_string({i: Z}, n)
    for i in range(n - 1):
        H += tensor_string({i: X, i + 1: X}, n)
    return H


def expectation(H, v):
    """矩阵 H 关于列向量 v 的期望"""
    return v.conj().T @ H @ v


if __name__ == "__main__":
    n = int(input("请输入量子比特数 n: "))

    H = build_H(n)
    v0 = np.zeros(2 ** n, dtype=complex)
    v0[0] = 1.0

    exp_val = expectation(H, v0)

    print(f"\nH 为 {H.shape[0]} x {H.shape[1]} 矩阵")
    print(f"H 在 (1,0,0,...) 下的期望值: {exp_val.real:.6f}  (理论值 = n = {n})")

    # n > 4 时矩阵太大 
    if n <= 4:
        np.set_printoptions(precision=4, suppress=True)
        print("\nH =")
        print(H.real)
