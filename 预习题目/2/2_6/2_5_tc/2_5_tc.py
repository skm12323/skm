# 2-5 狄拉克符号 —— 使用 TensorCircuit 后端 (jax) 重写
import numpy as np
import tensorcircuit as tc
from functools import reduce

K = tc.set_backend("jax")


def _np(t):
    """后端张量 -> numpy（仅用于打印 / allclose 对照）。"""
    return K.numpy(t)


# ---------- 单比特基矢 ----------

ket0 = K.convert_to_tensor(np.array([[1], [0]], dtype=complex))   # |0>
ket1 = K.convert_to_tensor(np.array([[0], [1]], dtype=complex))   # |1>
bra0 = K.transpose(K.conj(ket0))                                  # <0|
bra1 = K.transpose(K.conj(ket1))                                  # <1|


def kron_all(*mats):
    """多个矩阵/列向量的张量积（左为高位 q0）。"""
    return reduce(K.kron, mats)


def pauli_from_dirac():
    """用外积（Dirac 形式）构造泡利矩阵。"""
    I = K.matmul(ket0, bra0) + K.matmul(ket1, bra1)              # |0><0| + |1><1|
    X = K.matmul(ket0, bra1) + K.matmul(ket1, bra0)              # |0><1| + |1><0|
    Y = -1j * K.matmul(ket0, bra1) + 1j * K.matmul(ket1, bra0)   # -i|0><1| + i|1><0|
    Z = K.matmul(ket0, bra0) - K.matmul(ket1, bra1)              # |0><0| - |1><1|
    return I, X, Y, Z


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("=== 单比特基矢 ===")
    print("|0> = ", _np(ket0))
    print("|1> = ", _np(ket1))

    print("\n=== 泡利矩阵的 Dirac 外积形式 ===")
    I, X, Y, Z = pauli_from_dirac()
    X_std = K.stack([K.stack([0, 1]), K.stack([1, 0])])
    Y_std = K.stack([K.stack([0, -1j]), K.stack([1j, 0])])
    Z_std = K.stack([K.stack([1, 0]), K.stack([0, -1])])
    print("I = |0><0| + |1><1| =\n", _np(I))
    print("X = |0><1| + |1><0|    ", np.allclose(_np(X), X_std))
    print("Y = -i|0><1| + i|1><0| ", np.allclose(_np(Y), Y_std))
    print("Z = |0><0| - |1><1|    ", np.allclose(_np(Z), Z_std))

    print("\n=== 期望（2-3）===")
    theta = np.pi / 3
    v = K.matmul(K.expm(1j * theta * X / 2), ket0)               # v(θ) = e^{iθP/2}|0>
    expect = K.matmul(K.matmul(K.transpose(K.conj(v)), Z), v)     # <v|Z|v>  -> (1,1)
    val = _np(expect)[0, 0]
    print(f"<0|e^{{-iθX/2}} Z e^{{iθX/2}}|0> (θ=π/3) = {val.real:.6f}")

    print("\n=== 多比特基矢 |q0 q1 q2> ===")
    for bits in ["000", "010", "101", "111"]:
        k = kron_all(*[ket0 if b == "0" else ket1 for b in bits])
        idx = int(bits, 2)
        print(f"|{bits}> -> index {idx}")

    print("\n=== (|010> - |101>)/√2 的列向量 ===")
    ket010 = kron_all(ket0, ket1, ket0)
    ket101 = kron_all(ket1, ket0, ket1)
    psi = (ket010 - ket101) / np.sqrt(2)
    print("ψ =\n", _np(psi).flatten())
    norm = K.matmul(K.transpose(K.conj(psi)), psi)               # <ψ|ψ> -> (1,1)
    print("归一化验证 <ψ|ψ> =", _np(norm)[0, 0].real)
