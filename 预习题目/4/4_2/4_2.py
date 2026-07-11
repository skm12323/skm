import numpy as np
import tensorcircuit as tc

c = tc.Circuit(2)
c.h(0)
c.cx(0, 1)

# 测量
N = int(input("输入测量次数: "))
s = np.array(c.sample(batch=N, format="sample_int"))

compute_expectation = 0
for i in range(N):
    if s[i] == 3 or s[i] == 0:
        compute_expectation += 1/N
    else:
        compute_expectation -= 1/N

print("计算得到的期望: ", compute_expectation)