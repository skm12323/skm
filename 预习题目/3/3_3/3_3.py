import numpy as np
import tensorcircuit as tc

K = tc.set_backend("jax")

sigma_x = K.stack([K.stack([0, 1]), K.stack([1, 0])])
sigma_y = K.stack([K.stack([0, -1j]), K.stack([1j, 0])])
sigma_z = K.stack([K.stack([1, 0]), K.stack([0, -1])])
I_2 = K.stack([K.stack([1, 0]), K.stack([0, 1])])

def f(theta, P_1, P_2):
    exp_1 = K.expm(- 1j * theta * P_1 / 2)
    exp_2 = K.expm(1j * theta * P_1 / 2)
    P = K.matmul(exp_1, K.matmul(P_2, exp_2))
    v_0 = K.stack([1, 0])
    return K.tensordot(K.conj(v_0), K.matmul(P, v_0))

