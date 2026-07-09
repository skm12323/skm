import numpy as np
import tensorcircuit as tc

tc.set_dtype("complex128")

K = tc.set_backend("jax")

sigma_x = K.stack([K.stack([0, 1]), K.stack([1, 0])])
sigma_y = K.stack([K.stack([0, -1j]), K.stack([1j, 0])])
sigma_z = K.stack([K.stack([1, 0]), K.stack([0, -1])])

def f(theta, P_1, P_2):
    exp_1 = K.expm(- 1j * theta * P_1 / 2)
    exp_2 = K.expm(1j * theta * P_1 / 2)
    P = K.matmul(exp_1, K.matmul(P_2, exp_2))
    v_0 = K.stack([1, 0])
    return K.tensordot(K.conj(v_0), K.tensordot(P, v_0, axes=1), axes=1)

def compute_grad_numerical(f, theta, P_1, P_2, h=1e-5):
    f_backward = f(theta - h, P_1, P_2)
    f_forward = f(theta + h, P_1, P_2)
    grad = (f_forward - f_backward) / (2 * h)
    return grad

def compute_grad_by_paramete_shift(f, theta, P_1, P_2):
    f_plus = f(theta + np.pi / 2, P_1, P_2)
    f_minus = f(theta - np.pi / 2, P_1, P_2)
    grad = (f_plus - f_minus) / 2
    return grad
'''
if __name__ == "__main__":
    theta = 0.57
    P_1 = sigma_x
    P_2 = sigma_y

    grad_numerical = compute_grad_numerical(f, theta, P_1, P_2)
    grad_parameter_shift = compute_grad_by_paramete_shift(f, theta, P_1, P_2)

    print("Numerical gradient:", grad_numerical)
    print("Parameter-shift gradient:", grad_parameter_shift)
    print("Difference:", np.abs(grad_numerical - grad_parameter_shift))
'''
'''
能使用参数平移, 原因: 

    f展开后的所有项均为sin(θ/2)cos(θ/2), sin(θ/2)^2, cos(θ/2)^2形式, 因此一定能够转化为f(x) = A sin(x + B) + C的形式 ( 因为e^(i θ/2 P)的展开形式 )
'''
if __name__ == "__main__":
    theta = float(input("请输入theta的值: "))
    for a in ['x', 'y', 'z']:
        for b in ['x', 'y', 'z']:
            P_1 = {'x': sigma_x, 'y': sigma_y, 'z': sigma_z}[a]
            P_2 = {'x': sigma_x, 'y': sigma_y, 'z': sigma_z}[b]

            grad_numerical = compute_grad_numerical(f, theta, P_1, P_2)
            grad_parameter_shift = compute_grad_by_paramete_shift(f, theta, P_1, P_2)

            print(f"对σ_{a}和σ_{b}分别计算:")
            print("数值微分: ", grad_numerical)
            print("参数平移: ", grad_parameter_shift)
            print("Difference: ", np.abs(grad_numerical - grad_parameter_shift))
            print()