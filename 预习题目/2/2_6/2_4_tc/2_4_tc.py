import numpy as np
import tensorcircuit as tc
from functools import reduce

K = tc.set_backend("jax")

I = K.stack([K.stack([1, 0]), K.stack([0, 1])])
X = K.stack([K.stack([0, 1]), K.stack([1, 0])])
Z = K.stack([K.stack([1, 0]), K.stack([0, -1])])

def tensor_string(spec, n):
    ops = [I] * n
    for i, mat in spec.items():
        ops[i] = mat
    return reduce(K.kron, ops)

def build_H(n):
    N = 2 ** n
    H = K.zeros((N, N))
    for i in range(n):
        H = H + tensor_string({i: Z}, n)
    for i in range(n - 1):
        H = H + tensor_string({i: X, i + 1: X}, n)
    return H

def expectation(H, v):
    return K.tensordot(K.conj(v), K.tensordot(H, v, axes=1), axes=1)

if __name__ == "__main__":
    n = int(input("请输入量子比特数 n: "))

    H = build_H(n)
    #v0 = K.zeros(2 ** n)
    #v0 = v0.at[0].set(1) # jax后端特有API，换后端报错
    v0 = K.convert_to_tensor([1.] + [0.] * (2 ** n - 1))

    exp_val = expectation(H, v0)

    print(f"\nH 为 {H.shape[0]} x {H.shape[1]} 矩阵")
    print(f"H 在 (1,0,0,...) 下的期望值: {exp_val.real:.6f}  (理论值 = n = {n})")

    # n > 4 时矩阵太大 
    if n <= 4:
        np.set_printoptions(precision=4, suppress=True)
        print("\nH =")
        print(H.real)
