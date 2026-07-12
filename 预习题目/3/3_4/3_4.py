import numpy as np
import tensorcircuit as tc

tc.set_dtype("complex128")
K = tc.set_backend("jax")

sigma_x = K.stack([K.stack([0, 1]), K.stack([1, 0])])
sigma_y = K.stack([K.stack([0, -1j]), K.stack([1j, 0])])
sigma_z = K.stack([K.stack([1, 0]), K.stack([0, -1])])

# 原函数
def f(theta, P_1, P_2):
    exp_1 = K.expm(- 1j * theta * P_1 / 2)
    exp_2 = K.expm(1j * theta * P_1 / 2)
    P = K.matmul(exp_1, K.matmul(P_2, exp_2))
    v_0 = K.stack([1, 0])
    return K.tensordot(K.conj(v_0), K.tensordot(P, v_0, axes=1), axes=1)


# 用参数平移计算梯度
def compute_grad_by_paramete_shift(f, theta, P_1, P_2):
    f_plus = f(theta + np.pi / 2, P_1, P_2)
    f_minus = f(theta - np.pi / 2, P_1, P_2)
    grad = (f_plus - f_minus) / 2
    return grad


# 梯度下降
if __name__ == "__main__":
    
    theta_init = float(input("请输入theta的初始值: "))
    Lambda = float(input("请输入学习率lambda的值: "))
    n = int(input("请输入迭代次数n的值: "))
    test_if_is_const = 0.

    for a in ['x', 'y', 'z']:
        for b in ['x', 'y', 'z']:
            P_1 = {'x': sigma_x, 'y': sigma_y, 'z': sigma_z}[a]
            P_2 = {'x': sigma_x, 'y': sigma_y, 'z': sigma_z}[b]
            theta = theta_init
            test_if_is_const = 0.
            for i in range(n):
                grad = compute_grad_by_paramete_shift(f, theta, P_1, P_2)
                theta -= Lambda * grad
                test_if_is_const += np.abs(grad)
            final = f(theta, P_1, P_2)
            print(f"对σ_{a}和σ_{b}计算:")
            if np.abs(test_if_is_const) <= 1.e-8:
                print("梯度为0, 函数为常数函数, 常数函数值为: ", f"{K.numpy(final).real:.6f}")
            else:
                print("最终结果: ", f"{K.numpy(final).real:.6f}")
            print("最终theta的值: ", f"{K.numpy(theta).real:.6f}")
            print()


