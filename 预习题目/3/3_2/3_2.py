import numpy as np

def compute_derivative(f, x, delta):
    derivative = 0
    tau = 2 * np.sin(delta)
    f_forward = f(x + delta)
    f_backward = f(x - delta)
    derivative = (f_forward - f_backward)/tau
    return derivative

if __name__ == "__main__":

    A = float(input("请输入 A 的值: "))
    B = float(input("请输入 B 的值: "))
    C = float(input("请输入 C 的值: "))

    delta = float(input("请输入步长的值: "))
    x = float(input("请输入 x 的值: "))

    def f(x):
        return A * np.sin(x + B) + C
    
    def theoretical_derivative(x):
        return A * np.cos(x + B)
    
    def diff(x):
        diff = compute_derivative(f, x, delta) - theoretical_derivative(x)
        return diff

    print(diff(x))

    print("\n=== 参数平移 vs 中心差分（在输入的 x 处，不同 δ）===")
    print(f"{'delta':>8} | {'参数平移误差':>14} | {'中心差分误差':>14}")
    ex = theoretical_derivative(x)
    for d in [0.01, 0.1, 1.0, np.pi / 2, 10, 100]:
        ps = compute_derivative(f, x, d)            # 参数平移：τ = 2 sin(δ)
        cd = (f(x + d) - f(x - d)) / (2 * d)        # 普通中心差分
        print(f"{d:8.3f} | {ps - ex:+14.2e} | {cd - ex:+14.2e}")
