import numpy as np
from scipy.linalg import expm

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

P = sigma_x
theta = np.pi/2

exp_val = expm(1j * theta * P)
compute_exp_val = np.cos(theta) * np.eye(2, dtype=complex) + 1j * np.sin(theta) * P

diff_exp = exp_val - compute_exp_val

print(diff_exp)