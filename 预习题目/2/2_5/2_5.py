import numpy as np
from scipy.linalg import expm

# ---------- 单比特基矢 ----------
ket0 = np.array([[1], [0]], dtype=complex)   # |0> = (1, 0)^T
ket1 = np.array([[0], [1]], dtype=complex)   # |1> = (0, 1)^T
bra0, bra1 = ket0.conj().T, ket1.conj().T    # <0|, <1|  (行向量)


def kron_all(*mats):
    """多个矩阵/列向量的张量积（左为高位 q0）。"""
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def pauli_from_dirac():
    """用外积（Dirac 形式）构造泡利矩阵。"""
    I = ket0 @ bra0 + ket1 @ bra1             # |0><0| + |1><1|
    X = ket0 @ bra1 + ket1 @ bra0             # |0><1| + |1><0|
    Y = -1j * ket0 @ bra1 + 1j * ket1 @ bra0  # -i|0><1| + i|1><0|
    Z = ket0 @ bra0 - ket1 @ bra1             # |0><0| - |1><1|
    return I, X, Y, Z


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("=== 单比特基矢 ===")
    print("|0> =\n", ket0)
    print("|1> =\n", ket1)

    print("\n=== 泡利矩阵的 Dirac 外积形式 ===")
    I, X, Y, Z = pauli_from_dirac()
    X_std = np.array([[0, 1], [1, 0]], dtype=complex)
    Y_std = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z_std = np.array([[1, 0], [0, -1]], dtype=complex)
    print("I = |0><0| + |1><1| =\n", I)
    print("X = |0><1| + |1><0|   与标准一致:", np.allclose(X, X_std))
    print("Y = -i|0><1| + i|1><0| 与标准一致:", np.allclose(Y, Y_std))
    print("Z = |0><0| - |1><1|   与标准一致:", np.allclose(Z, Z_std))

    print("\n=== 期望即 <v|Q|v>（对应 2-3）===")
    theta = np.pi / 3
    v = expm(1j * theta * X / 2) @ ket0       # v(θ) = e^{iθP/2}|0>
    expect = (v.conj().T @ Z @ v)[0, 0]       # <v|Z|v>
    print(f"<0|e^{{-iθX/2}} Z e^{{iθX/2}}|0> (θ=π/3) = {expect.real:.6f}")

    print("\n=== 多比特基矢 |q0 q1 q2>（左为高位，下标=二进制值）===")
    for bits in ["000", "010", "101", "111"]:
        k = kron_all(*[ket0 if b == "0" else ket1 for b in bits])
        idx = int(bits, 2)
        print(f"|{bits}> -> index {idx}  (核对非零位: {int(np.argmax(k))})")

    print("\n=== (|010> - |101>)/√2 的列向量 ===")
    ket010 = kron_all(ket0, ket1, ket0)
    ket101 = kron_all(ket1, ket0, ket1)
    psi = (ket010 - ket101) / np.sqrt(2)
    print("ψ =\n", psi.flatten())
    print("归一化 <ψ|ψ> =", (psi.conj().T @ psi)[0, 0].real)
    print("仅在 index 2 (|010>) 为 +1/√2, index 5 (|101>) 为 -1/√2, 其余为 0")
