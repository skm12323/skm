import numpy as np

theta = np.pi/4 #旋转角度（逆时针）

R = np.array([
    [np.cos(theta), - np.sin(theta)],
    [np.sin(theta),np.cos(theta)]
])

v = np.array([1., 0.])

v_1 = R @ v

print(v_1)