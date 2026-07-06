import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import os

def expectation(axis_1, axis_2, theta):
    pauli_matrices = {
        'x': np.array([[0, 1], [1, 0]], dtype=complex),
        'y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'z': np.array([[1, 0], [0, -1]], dtype=complex)
    }

    if axis_1 not in pauli_matrices or axis_2 not in pauli_matrices:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")

    P = pauli_matrices[axis_1]
    Q = pauli_matrices[axis_2]

    exp_val = expm(1j * theta * P / 2)
    v_0 = np.array([1, 0], dtype=complex)
    v = exp_val @ v_0
    expectation_value = v.conj().T @ Q @ v
    return expectation_value


if __name__ == "__main__":
    thetas = np.linspace(0, 2 * np.pi, 400)

    combos = [
        ("x", "x"), ("x", "y"), ("x", "z"),
        ("y", "x"), ("y", "y"), ("y", "z"),
        ("z", "x"), ("z", "y"), ("z", "z"),
    ]

    plt.figure(figsize=(9, 6))
    for a, b in combos:
        vals = [np.real(expectation(a, b, t)) for t in thetas]
        plt.plot(thetas, vals, label=f"rotate {a} -> expect {b}")

    plt.xlabel(r"$\theta$")
    plt.ylabel("Expectation value")
    plt.title("Pauli expectation vs theta for various axis combinations")
    plt.legend(ncol=3, fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pauli_expectations.png",
    )
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")
    plt.show()

