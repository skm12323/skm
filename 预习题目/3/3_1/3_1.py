import numpy as np


def compute_grad(f, x, option=1, skip_check=False, h=1e-5):
    """数值梯度。

    参数
    ----
    f        : 输入向量 -> 输出标量的函数
    x        : 求导点（一维向量）
    option   : 1 = 前向差分 (f(x+h)-f(x))/h   ← 题目给出的公式
               0 = 中心差分 (f(x+h)-f(x-h))/(2h)   ← 精度更高 O(h^2)
    skip_check: True = 跳过步长提示
    h        : 差分步长

    返回
    ----
    grad : 与 x 同形状的梯度向量
    """
    if option not in (0, 1):
        raise ValueError(f"option 只能取 0(中心) 或 1(前向)，得到 {option}")

    n = len(x)
    grad = np.zeros(n)
    f_center = f(x)

    for i in range(n):
        # 前向点
        x_forward = x.copy()
        x_forward[i] += h
        f_forward = f(x_forward)

        if option == 1:                   # 前向差分：(f(x+h)-f(x))/h
            grad[i] = (f_forward - f_center) / h
        else:                             # 中心差分：(f(x+h)-f(x-h))/(2h)
            x_backward = x.copy()
            x_backward[i] -= h
            f_backward = f(x_backward)
            grad[i] = (f_forward - f_backward) / (2 * h)

        # 可选：步长是否合适的提示
        if not skip_check:
            if abs(f_forward - f_center) < 1e-15:
                print(f"警告: 维度{i} 的函数值变化极小，步长 h = {h} 可能过小")
            if abs(f_forward - f_center) > 1e6 * abs(f_center) + 1e-6:
                print(f"警告: 维度{i} 的函数值变化剧烈，步长 h = {h} 可能过大")

    return grad


# 示例使用
if __name__ == "__main__":
    # 总开关：0 = 用默认值直接输出；1 = 每个测试前交互选择
    interactive = int(input("是否进行具体选择？(0=直接输出, 1=每次选择): "))

    def ask_options():
        """交互式获取每次计算的选项。"""
        option = int(input("  差分方式 (1=前向, 0=中心): "))
        skip   = int(input("  跳过步长检查？(1=跳过, 0=检查): "))
        h      = float(input("  步长 h (如 1e-5): "))
        return {"option": option, "skip_check": bool(skip), "h": h}

    DEFAULT = {"option": 1, "skip_check": False, "h": 1e-5}   # 题目公式：前向差分

    # 测试函数1：正常的二次函数
    def f1(x):
        return x[0]**2 + x[1]**2

    x1 = np.array([1.0, 2.0])
    kw = ask_options() if interactive else DEFAULT
    print("=== f1 = x0^2 + x1^2 ===")
    print("梯度:", compute_grad(f1, x1, **kw), " 理论值: [2.0, 4.0]\n")

    # 测试函数2：变化剧烈的函数（h 偏大会触发警告）
    def f2(x):
        return np.exp(100 * x[0]) - 1

    x2 = np.array([0.0])
    kw = ask_options() if interactive else {**DEFAULT, "h": 1e-3}
    print("=== f2 = exp(100 x0) - 1 ===")
    print("梯度:", compute_grad(f2, x2, **kw), " 理论值: [100.0]\n")

    # 测试函数3：非常平坦的函数（h 偏小会触发警告）
    def f3(x):
        return 1e-20 * x[0]**2

    x3 = np.array([1.0])
    kw = ask_options() if interactive else {**DEFAULT, "h": 1e-8}
    print("=== f3 = 1e-20 x0^2 ===")
    print("梯度:", compute_grad(f3, x3, **kw), " 理论值: [2.e-20]")
